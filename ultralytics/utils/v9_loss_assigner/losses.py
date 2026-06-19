import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.tal import Dense_TaskAlignedAssigner, bbox2dist
from ultralytics.utils.metrics import bbox_iou

def nwd_loss(pred_bboxes, target_bboxes, eps=1e-6, constant=12.8):
    """
    Normalized Wasserstein Distance (NWD) Loss for tiny objects.
    ISPRS 2021 theory based.
    """
    p_cx, p_cy, p_w, p_h = pred_bboxes.chunk(4, dim=-1)
    t_cx, t_cy, t_w, t_h = target_bboxes.chunk(4, dim=-1)

    # Calculate Wasserstein distance between bounded boxes modeling them as 2D Gaussians
    w_dist = ((p_cx - t_cx)**2 + (p_cy - t_cy)**2 + 
              ((p_w/2) - (t_w/2))**2 + ((p_h/2) - (t_h/2))**2)
    
    nwd = torch.exp(-torch.sqrt(w_dist + eps) / constant)
    return 1 - nwd

def repulsion_loss(pred_bboxes, target_bboxes, iou_threshold=0.5):
    """
    Repulsion Loss for Crowded Scenes.
    CVPR 2018 theory based.
    Pushes bounding boxes away from non-target neighboring ground truths.
    """
    # Simplification for speed: compute pairwise IoU and penalize
    # Calculate memory-efficient sparse pairwise IoUs or chunk it
    # Simplified repulsion: only penalize cross-IoU between pred and target within a batch
    # To prevent OOM, just use diagonal and a random shifted subset
    n = pred_bboxes.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred_bboxes.device)
        
    shifted_targets = torch.roll(target_bboxes, shifts=1, dims=0)
    ious = bbox_iou(pred_bboxes, shifted_targets, xywh=False)  # Shape (N) not (N,M) Tensor
    
    penalty = torch.clamp(ious - iou_threshold, min=0.0)
    return penalty.mean()

class NWD_Repulsion_BboxLoss(BboxLoss):
    def __init__(self, reg_max=16):
        super().__init__(reg_max=reg_max)
        self.reg_max = reg_max
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Modified Bbox Loss combining CIoU, NWD, and Repulsion."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # Original IoU / DFL
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # NWD for small objs
        loss_nwd = (nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]) * weight).sum() / target_scores_sum

        # Repulsion for crowdedness
        if fg_mask.sum() > 1:
            loss_rep = repulsion_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        else:
            loss_rep = 0.0

        # DFL
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou + 0.5 * loss_nwd + 0.5 * loss_rep, loss_dfl


class NWD_Repulsion_DetectionLoss(v8DetectionLoss):
    """
    V9 Extended Detection Loss:
    Integrates Density-Guided TaskAlignedAssigner and NWD/Repulsion Loss.
    """
    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk)
        # Override the assigner to our Density-Guided Version
        self.assigner = Dense_TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        # Override bbox loss
        m = model.model[-1]
        self.bbox_loss = NWD_Repulsion_BboxLoss(m.reg_max).to(self.device)

