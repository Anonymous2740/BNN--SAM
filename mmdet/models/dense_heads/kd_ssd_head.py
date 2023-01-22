from mmdet.models.losses.kl_loss import loss_kl
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

import mmcv
from mmdet.core import get_classes
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

from ..builder import build_loss


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    # model = nn.DataParallel(model, device_ids=[0, 1])
    # model.to(device)
    model = model.cuda()
    model.to(torch.cuda.current_device())
    model.eval()
    return model

# TODO: add loss evaluator for SSD
@HEADS.register_module()
class KDSSDHead(AnchorHead):
    """SSD head with Knowledge distillation used in https://arxiv.org/abs/1512.02325.

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
                 test_cfg=None,
                 teacher_config='configs/pascal_voc/ssd300_voc0712.py',
                 teacher_model='experiments/ssd_no_binary/epoch_24.pth',
                 kl_loss = dict(type='kl_loss')
                 ):
        super(AnchorHead, self).__init__()

        self.teacher_models = init_detector(teacher_config, teacher_model)

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
        self.GHMC = GHMC(bins=10, momentum=0, use_sigmoid=True)
        self.GHMR = GHMR(mu=0.02, bins=10, momentum=0, loss_weight=1.0)

        self.loss_dfl = build_loss(kl_loss)

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
        feats_list = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

            feats_list.append(feat)

        return feats_list, cls_scores, bbox_preds


    def loss_single(self, gt_bboxes, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights,  soft_targets, num_total_samples):
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
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights 
        # loss_cls_all = self.LabelSmoothingCrossEntropy(cls_score, labels)
        # Focal_cost = self.focal_loss(cls_score, labels)
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)
        # loss_cls = self.GHMC( cls_score, labels, label_weights)


        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.num_classes)).nonzero().reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero().view(-1)

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

        kl_loss = loss_kl("量化模型的输出结果",soft_targets)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        

        
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             feats_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             img,
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

        with torch.no_grad():
            soft_target = self.teacher_models(
                return_loss = False,
                rescale = True,
                img = [img],
                img_metas = [img_metas])[1]
            

        cls_reg_targets = self.get_targets(
            all_cls_scores,
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

        soft_target = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in soft_target
        ], -2)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # check NaN and Inf
        assert torch.isfinite(all_cls_scores).all().item(), \
            'classification scores become infinite or NaN!'
        assert torch.isfinite(all_bbox_preds).all().item(), \
            'bbox predications become infinite or NaN!'


        losses_cls, losses_bbox= multi_apply(
            self.loss_single,
            gt_bboxes,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            soft_target,
            num_total_samples=num_total_pos,
            )

 

        return dict(loss_cls = losses_cls,loss_bbox = losses_bbox) #,loss_p = loss_p)#,loss_cosin = loss_cosin)# loss_pearson = loss_pearson)
        

