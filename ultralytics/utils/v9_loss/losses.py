import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist
from ultralytics.utils.metrics import bbox_iou

class Dense_TaskAlignedAssigner(TaskAlignedAssigner):
    """Density-Guided Task Aligned Assigner."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        # Fallback to base but wrap if density logic is expanded further
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        # Placeholder for Density Penality logic in dense bounding box assignments
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

def nwd_loss(pred_bboxes, target_bboxes, eps=1e-6, constant=12.8):
    p_cx, p_cy, p_w, p_h = pred_bboxes.chunk(4, dim=-1)
    t_cx, t_cy, t_w, t_h = target_bboxes.chunk(4, dim=-1)
    w_dist = ((p_cx - t_cx)**2 + (p_cy - t_cy)**2 + 
              ((p_w/2) - (t_w/2))**2 + ((p_h/2) - (t_h/2))**2)
    nwd = torch.exp(-torch.sqrt(w_dist + eps) / constant)
    return 1 - nwd

def repulsion_loss(pred_bboxes, target_bboxes, iou_threshold=0.5):
    ious = bbox_iou(pred_bboxes, target_bboxes, squeeze=False)
    mask = ~torch.eye(ious.shape[0], dtype=torch.bool, device=ious.device)
    if ious.dim() > 2:
        mask = mask.unsqueeze(0).expand(ious.shape[0], -1, -1)
    rep_ious = ious * mask
    penalty = torch.clamp(rep_ious - iou_threshold, min=0.0)
    return penalty.mean()

class NWD_Repulsion_BboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        loss_nwd = (nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / target_scores_sum
        
        if fg_mask.sum() > 1:
            loss_rep = repulsion_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        else:
            loss_rep = 0.0

        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou + 0.5 * loss_nwd + 0.5 * loss_rep, loss_dfl

class NWD_Repulsion_DetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        self.assigner = Dense_TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        m = model.model[-1]
        self.bbox_loss = NWD_Repulsion_BboxLoss(m.reg_max).to(self.device)
