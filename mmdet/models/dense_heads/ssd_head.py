from mmdet.core.bbox import iou_calculators
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
# from ..losses import balanced_l1_loss
from .anchor_head import AnchorHead
import numpy

from ..losses import FocalLoss,LabelSmoothingCrossEntropy,GHMC,GHMR

from scipy.stats import pearsonr
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

import numpy as np
from numpy import nan
from sklearn.cluster import KMeans

from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, paired_distances



# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied on decoded bounding boxes. Default: False
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes + 1  # add background class
        self.anchor_generator = build_anchor_generator(anchor_generator)
        num_anchors = self.anchor_generator.num_base_anchors
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        feat_size = (38,19,10,5,3,1)
        reg_convs = []
        cls_convs = []
        for i in range(len(in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * 4, # Oringinal version is 4,Bi-det is 8
                    kernel_size=3,
                    padding=1))
            cls_convs.append(
                nn.Conv2d(
                    in_channels[i],
                    num_anchors[i] * (num_classes + 1),
                    kernel_size=3,
                    padding=1))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # set sampling=False for archor_target
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False
        self.softmax = torch.nn.Softmax(dim=-1)
        self.LabelSmoothingCrossEntropy = LabelSmoothingCrossEntropy()
        # self.GHMC = GHMC(bins=10, momentum=0, use_sigmoid=True)
        # self.GHMR = GHMR(mu=0.02, bins=10, momentum=0, loss_weight=1.0)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        # feats_list = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

            # feats_list.append(feat)

        return  cls_scores, bbox_preds
    

    def loss_single(self, gt_bboxes, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights,  num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        
        # reweight_loss = False
        # if reweight_loss == True:
        #     label_weights = label_weights.clone().detach()

        #     V_appearance = []

        #     conf_score = self.softmax(cls_score).max(axis=-1)[0]
        #     # (key, value): (index, score)
        #     k_score_idx = np.argsort((-conf_score).cpu().data) # 把conf_score从大到小排序
        #     # v_score = np.sort((-conf_score).cpu().data) 
            
        #     # core_box_index = k_score_idx[0]  # the max conf score of the bboxes
        #     bbox_pred_new = self.bbox_coder.decode(anchor, bbox_pred) 
 
            
        #     keep = conf_score.new(conf_score.size(0)).zero_().long()
        #     count = 0
        #     while k_score_idx.numel() > 0:
        #         i = k_score_idx[0]
        #         keep[count] = i
        #         count += 1
        #         if k_score_idx.size(0) == 1:
        #             break
        #         k_score_idx = k_score_idx[1:] # remove kept element from view
        #         IOU = bbox_overlaps(bbox_pred_new[i].reshape(1,-1).expand(2,-1).cpu().detach().numpy(),bbox_pred_new[k_score_idx].cpu().detach().numpy())[0]
        #         # keep only elements with IOU <= 0.00001
        #         k_score_idx = k_score_idx[IOU <= 0.00001]
            
        #     label_weights *= 1.0
        #     label_weights[keep] = 2.0
             
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights 


        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred) 

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)

        # loss_bbox = balanced_l1_loss(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     beta=1.0,
        #     alpha=0.5,
        #     gamma=1.5,
        #     reduction='mean')
        return loss_cls[None], loss_bbox
        
    def log_func(self, tensor):
        return tensor * torch.log(tensor)
    def nms(self,boxes, scores, overlap=0.5, top_k=200):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """

        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]

        return keep, count
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             mutual_guide=True
             ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(#anchor_list  [num_images,6,the feature level anchors,4]  the feature level anchors, including [5776,2166,600,150,36,4]
            featmap_sizes, img_metas, device=device)
        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape( # s.permute() 交换s的维度
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)

       
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=False)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
# loss_r

        REGULARIZATION_LOSS_WEIGHT = 0.0 # 0.1
        if REGULARIZATION_LOSS_WEIGHT != 0.:
            f_num = len(feats_list) 
            loss_r = 0.

            for f_m in feats_list:
                loss_r += (f_m ** 2).mean()

            
            loss_r *= REGULARIZATION_LOSS_WEIGHT
            #loss_r /= float(f_num)
            loss_r /= (float(f_num) + 1e-6)  




        all_labels = torch.cat(labels_list, -1).view(num_images, -1)


        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)




        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'

        #The loss p (the information entropy of confidence score)
        conf_data = self.softmax(all_cls_scores)
        batch_size = num_images
        num_priors = all_bbox_preds.size(1)
        self.top_k = 200
        self.conf_thresh = 0.02 #0.03
        output = torch.zeros(batch_size, self.cls_out_channels, self.top_k, 5)
        conf_preds = conf_data.view(batch_size, num_priors,
                                    self.cls_out_channels).transpose(2, 1)


        PRIOR_LOSS_WEIGHT =  0.0 #0.2
        loss_p = 0.

        if PRIOR_LOSS_WEIGHT != 0.:
            loss_count = 0.

            # Decode predictions into bboxes.
            for i in range(batch_size):
                decoded_boxes = self.bbox_coder.decode(all_anchors[i],all_bbox_preds[i],max_shape=[300,300])
                scale_factor = img_metas[i]['scale_factor']
                decoded_boxes /= decoded_boxes.new_tensor(scale_factor)
                # For each class,perform nms
                conf_scores = conf_preds[i].clone()

                # for cl in range(1, self.cls_out_channels):
                for cl in range(0, self.cls_out_channels-1):
                    # perform nms only for gt class when calculating prior loss
                    if gt_labels is not None:
                        if not (gt_labels[i] == cl).any():
                            continue
                    c_mask = conf_scores[cl].gt(self.conf_thresh)   
                    scores = conf_scores[cl][c_mask] 
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = self.nms(boxes.data, scores.data, 0.45, self.top_k) #self.nms_thresh=0.45
                    output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                    boxes[ids[:count]]), 1) # output shape [15,21,200,5]


            # skip j = self.cls_out_channels, because it's the background class
            # for j in range(1,self.cls_out_channels):
            for j in range(0, self.cls_out_channels-1):
                all_dets = output[:, j, :, :] 
                all_mask = all_dets[:, :, :1].gt(0.).expand_as(all_dets)  # [batch, top_k, 5]

                for batch_idx in range(batch_size):   
                    # skip non-existed class         
                    if not (gt_labels[batch_idx] == j ).any():
                        continue  

                    dets = torch.masked_select(all_dets[batch_idx], all_mask[batch_idx]).view(-1, 5)  # [num, 5]
                    
                    if dets.size(0) == 0:
                        continue

                    # if pred num == gt num, skip 
                    if dets.size(0) <= ((gt_labels[batch_idx] == j ).sum().detach().cpu().item()):
                        continue

                    scores = dets[:, 0]  # [num]
                    scores_sum = scores.sum().item()  # no grad
                    scores = scores / scores_sum  # normalization
                    log_scores = self.log_func(scores)
                    gt_num = (gt_labels[batch_idx] == j ).sum().detach().cpu().item()
                    loss_p += (-1. * log_scores.sum() / float(gt_num))
                    loss_count += 1.
  
            loss_p /= (loss_count + 1e-6)
            loss_p *= PRIOR_LOSS_WEIGHT    
        loss_p = torch.Tensor([loss_p])   
        loss_p = loss_p.cuda() 

        loss_pearson_count = 0.
        loss_pearson = 0.0 
        PEARSON_LOSS_WEIGHT = 0.0 #0.0
        if PEARSON_LOSS_WEIGHT != 0.:
            for i in range(batch_size):                
                # Decode predictions into bboxes.
                decoded_boxes = self.bbox_coder.decode(all_anchors[i],all_bbox_preds[i],max_shape=[300,300])
                scale_factor = img_metas[i]['scale_factor']
                decoded_boxes /= decoded_boxes.new_tensor(scale_factor)

                conf_scores = conf_preds[i].clone()

                for cl in range(0, self.cls_out_channels-1):
                    if gt_labels is not None:
                        if not (gt_labels[i] == cl).any():
                            continue
                    c_mask = conf_scores[cl].gt(self.conf_thresh)   
                    scores = conf_scores[cl][c_mask] 
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = self.nms(boxes.data, scores.data, 0.45, self.top_k)#self.nms_thresh=0.45

                    output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                    boxes[ids[:count]]), 1)


            for j in range(0,self.cls_out_channels-1):
                all_dets = output[:, j, :, :] 

                for batch_idx in range(batch_size):  
                    dets = all_dets[batch_idx]          
                    if not (gt_labels[batch_idx] == j ).any():
                        # non-existed class
                        ious_max_per_image=np.zeros(len(dets))
                    else:
                        index_gt_labels = [x for (x,m) in enumerate(gt_labels[batch_idx].tolist()) if m == j]            
                        ious_per_image = bbox_overlaps(dets[:,1:].detach().numpy(),gt_bboxes[batch_idx][index_gt_labels].view(-1,4).cpu().detach().numpy())
                        ious_max_per_image = ious_per_image.max(axis=1) # dets和哪个GT的iou最大，就让它们Match起来

                    simi = np.mean(abs(dets[:,0].detach().numpy()-ious_max_per_image))

                    loss_pearson += simi
                    loss_pearson_count += 1.
            loss_pearson /= (loss_pearson_count + 1e-6)
            loss_pearson *= PEARSON_LOSS_WEIGHT
        loss_pearson = torch.Tensor([loss_pearson])  
        loss_pearson = loss_pearson.cuda()   


        loss_cosin_count = 0.
        loss_cosin = 0.
        COSIN_LOSS_WEIGHT = 0.0 #10.0 #3.0 #0.0 #0.2
        if COSIN_LOSS_WEIGHT != 0.:
            for i in range(batch_size):
                
                decoded_boxes = self.bbox_coder.decode(all_anchors[i],all_bbox_preds[i],max_shape=[300,300])
                conf_scores = conf_preds[i].clone()
                scale_factor = img_metas[i]['scale_factor']
                decoded_boxes /= decoded_boxes.new_tensor(scale_factor)


                for cl in range(0, self.cls_out_channels-1):
                    if gt_labels is not None:
                        if not (gt_labels[i] == cl).any():
                            continue
                    c_mask = conf_scores[cl].gt(self.conf_thresh)   
                    scores = conf_scores[cl][c_mask] 
                    if scores.size(0) == 0:
                        continue
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, count = self.nms(boxes.data, scores.data, 0.45, self.top_k) # self.nms_thresh=0.45
                    output[i, cl, :count] = \
                        torch.cat((scores[ids[:count]].unsqueeze(1),
                                    boxes[ids[:count]]), 1)#output shape [15,21,200,5]
            for j in range(0,self.cls_out_channels-1):
                all_dets = output[:, j, :, :] 
                all_mask = all_dets[:, :, :1].gt(0.).expand_as(all_dets)  # [batch, top_k, 5]
                

                # Classification to Location
                for batch_idx in range(batch_size):            
                    if not (gt_labels[batch_idx] == j ).any():
                        continue   
                    dets = torch.masked_select(all_dets[batch_idx], all_mask[batch_idx]).view(-1, 5)  # [num, 5]
                    if dets.size(0) == 0:
                        continue
                    # if pred num == gt num, skip 
                    if dets.size(0) <= ((gt_labels[batch_idx] == j ).sum().detach().cpu().item()):
                        continue
                    # the Pearson correlation coefficient between cls_score and IOU
                    index_gt_labels = [x for (x,m) in enumerate(gt_labels[batch_idx].tolist()) if m == j]
                    

                    X = dets[:,1:]
                    kmeans = KMeans(n_clusters=len(index_gt_labels), random_state=0).fit(X.detach().numpy())
                    
                    for m in range(len(kmeans.cluster_centers_)):
                        for n in range(len(kmeans.labels_)):
                            if kmeans.labels_[n] == m :
                                # simi = bbox_overlaps(dets[:,1:][n].detach().numpy(),kmeans.cluster_centers_[m])
                                # np.dot 点积
                                # np.linalog.norm 向量的2范数, 即向量的每个元素的平方和再开平方根，
                                if (np.linalg.norm(dets[:,1:][n].detach().numpy())*np.linalg.norm(kmeans.cluster_centers_[m])) != 0:
                                    simi = np.dot(dets[:,1:][n].detach().numpy(),kmeans.cluster_centers_[m])/(np.linalg.norm(dets[:,1:][n].detach().numpy())*np.linalg.norm(kmeans.cluster_centers_[m]))
                                    dist = 1-simi
                                    loss_cosin += dist
                                    loss_cosin_count += 1
            loss_cosin /= (loss_cosin_count + 1e-6)
            loss_cosin *= COSIN_LOSS_WEIGHT
        loss_cosin = torch.Tensor([loss_cosin])  
        loss_cosin = loss_cosin.cuda()                                 



        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            gt_bboxes,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            )

        return dict(loss_cls = losses_cls, loss_bbox = losses_bbox) #,loss_p = loss_p)#,loss_cosin = loss_cosin)# 
        




