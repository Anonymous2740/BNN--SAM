import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
import torch.nn.functional as F

@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def mutual_match(
        self,
        truths,
        priors,
        regress,
        classif,
        labels,
        loc_t,
        conf_t,
        overlap_t,
        pred_t,
        idx,
        ):
        """Classify to regress and regress to classify, Mutual Match for label assignement.
        Args:
            truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
            priors: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4].
            regress: (tensor) Regression prediction, Shape: [num_priors, 4].
            classif: (tensor) Classification prediction, Shape: [num_priors, num_classes].
            labels: (tensor) All the class labels for the image, Shape: [num_obj].
            loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
            conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
            overlap_t: (tensor) Tensor to be filled w/ iou score for each priors.
            pred_t: (tensor) Tensor to be filled w/ pred score for each priors.
            idx: (int) current batch index
        """

        num_obj = truths.size()[0]
        acr_overlaps = self.iou_calculator(truths, priors) #IOUanchor shape: [num_objects, num_priors]
        reg_overlaps = self.iou_calculator(truths, regress) #IOUregresed

        self.softmax = torch.nn.Softmax(dim=-1)
        # conf_.sdata = selfoftmax(all_cls_scores).t()[gt_labels,:]
        # pred_classifs = self.softmax(classif).t()[labels,:]
        pred_classifs = classif.sigmoid().t()[labels,:]
       
        sigma = 2.0
        pred_classifs = acr_overlaps ** ((sigma - pred_classifs) / sigma)#根据分类分数来正比例增大IOU,sigma为增大因子

        # pred_classifs = sigma * acr_overlaps / (sigma-pred_classifs)
         # ## at least 1 anchor per object ###

        if num_obj != 0:#所有的图像
            acr_overlaps[torch.arange(num_obj), acr_overlaps.max(1)[1]] = 1.0         #从GroundTruth角度出发，每个GT需对应一个与其交并比最大的1 anchor，找到其索引，把它设置为1
            reg_overlaps[torch.arange(num_obj), reg_overlaps.max(1)[1]] = 1.0
            pred_classifs[torch.arange(num_obj), pred_classifs.max(1)[1]] = 1.0
            acr_overlaps[acr_overlaps != acr_overlaps.max(dim=0,#从prior出发，每一个priors对应一个与其交并比最大的GT，小的设置为0
                        keepdim=True)[0]] = 0.0 
            reg_overlaps[reg_overlaps != reg_overlaps.max(dim=0,
                        keepdim=True)[0]] = 0.0
            pred_classifs[pred_classifs != pred_classifs.max(dim=0,
                        keepdim=True)[0]] = 0.0

            # ## assign pos and ign nums according to acr_overlaps ###

            #设置两个阈值用于标记正样本、负样本和忽略样本（以0.5，0.4为阈值）
            for (reg_overlap, pred_classif, acr_overlap) in zip(reg_overlaps,
                    pred_classifs, acr_overlaps):
                num_ign = (acr_overlap >= 0.4).sum() #忽略样本的阈值应该在0.4和0.5之间？这里是不是写错了！！
                num_pos = (acr_overlap >= 0.5).sum()

                # Location to classify
                # IOUregresses的阈值设计依然参考了IOUanchor
                # 负样本采用难例挖掘
                ign_mask = torch.topk(reg_overlap, num_ign, largest=True)[1]
                pos_mask = torch.topk(reg_overlap, num_pos, largest=True)[1] #
                reg_overlap[ign_mask] = 2.0 
                reg_overlap[pos_mask] = 3.0

                pos_mask = torch.topk(pred_classif, num_pos, largest=True)[1]
                pred_classif[pos_mask] = 3.0

                # ## for classification ###

                # reg_overlaps是预设的anchor box经过回归调整位置之后的box,与GTBox之间的IOU得分  即Localize to classify
                # shape : [[num_objects, num_regbox]]
                (best_truth_overlap, best_truth_idx) = reg_overlaps.max(dim=0) #每一列
                overlap_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
                conf_t[idx] = labels[best_truth_idx]  # [num_priors] top class label for each prior

                # ## for regression ###
                # Classsify to localize
                (best_truth_overlap, best_truth_idx) = pred_classifs.max(dim=0)
                pred_t[idx] = best_truth_overlap  # [num_priors] jaccord for each prior
                loc_t[idx] = truths[best_truth_idx]  # Shape: [num_priors,4] #为什么放的是truths的信息



    def assign(self,  bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):#all_cls_scores
        """Assign gt to bboxes.匹配gt和bboxes

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.
        
        该方法会对一个gt bbox匹配每一个bbox,每个bbox会分配为-1，-1代表是负样本；其余非负样本，会分配一个对应的gt的索引

        1. assign every bbox to the background  首先初始化时候假设每个anchor的mask都是-1，表示都是忽略anchor
        2. assign proposals whose iou with all gts < neg_iou_thr to 0  将每个anchor和所有gt的iou的最大Iou小于neg_iou_thr的anchor的mask设置为0，表示是负样本(背景样本)
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox  对于每个anchor，计算其和所有gt的iou，选取最大的iou对应的gt位置，如果其最大iou大于等于pos_iou_thr，则设置该anchor的mask设置为1，表示该anchor负责预测该gt bbox,是高质量anchor
        4. for each gt bbox, assign its nearest proposals (may be more than one) to itself
           3的设置可能会出现某些gt没有分配到对应的anchor(由于iou低于pos_iou_thr)，故下一步对于每个gt还需要找出和最大iou的anchor位置，如果其iou大于min_pos_iou，将该anchor的mask设置为1，表示该anchor负责预测对应的gt。通过本步骤，可以最大程度保证每个gt都有anchor负责预测，如果还是小于min_pos_iou，那就没办法了，只能当做忽略样本了。从这一步可以看出，3和4有部分anchor重复分配了，
           即当某个gt和anchor的最大iou大于等于pos_iou_thr，那肯定大于min_pos_iou，此时3和4步骤分配的同一个anchor。

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            # all_cls_scores = all_cls_scores.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()


        
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        
        # self.softmax = torch.nn.Softmax(dim=-1)
        # conf_data = self.softmax(all_cls_scores).t()[gt_labels,:]

        # # alpha = 0.15
        # # overlaps = (1-alpha)* overlaps + alpha * conf_data


        # sigma = 2.0
        
        # mask1 = (conf_data < 0.1).type(torch.float32)#0.1
        # conf_data_min = mask1 * conf_data
        # # overlaps_min = overlaps** (1/(2*conf_data_min))#
        # # 从0.1修正到了1
        # overlaps_min = overlaps ** (sigma/(sigma - (0.1-conf_data_min))) # overlaps_min = overlaps ** (2/(2-(1-score))) overlaps
        # overlaps_min = overlaps_min * mask1

        # mask2 =(conf_data>=0.1).type(torch.float32)#0.1
        # conf_data_max = mask2 * conf_data # conf_data_max=0时，对应的overlaps也不等于0
        # overlaps_max = overlaps ** ((sigma - conf_data_max) / sigma)
        # overlaps_max = overlaps_max * mask2

        # overlaps = overlaps_min + overlaps_max

        

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        # 1. 所有index全部设置为-1，表示全部是忽略anchor
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # 计算每个anchor,和那个gt的iou最大
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        #计算每个gt,和哪个anchor的iou最大,可能两个max的索引有重复
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        # 2. 对于每个anchor,计算其和gt的最大iou都小于neg_iou_thr阈值，则分配负样本
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        # 3. 对于每个anchor,计算其和gt的最大iou大于pos_iou_thr阈值，则分配正样本
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            # 4.对于每个gt,如果其和某个anchor的最大iou大于min_pos_iou阈值，那么依然分配正样本
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    #该参数的含义是：当某个gt,和其中好几个anchor都是最大iou(最大iou对应的anchor有好几个时候)，则全部分配正样本
                    #该操作可能出现某几个anchor和同一个Gt匹配，都负责预测
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        #仅仅考虑最大的一个，不考虑多个最大时候
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
# num_gts,           GT的数量
# assigned_gt_inds,  对于每一个anchor，和其有最大IOU的GT的inds(索引)
# max_overlaps,      计算每个anchor,和哪个gt的iou最大
