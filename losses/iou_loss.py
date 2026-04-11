"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
        if self.reduction not in {"none","mean","sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.
        px, py, pw, ph = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
        tx, ty, tw, th = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]

        # convert to corner format (x1, y1, x2, y2)
        px1 = px - pw/2
        py1 = py - ph/2
        px2 = px + pw/2
        py2 = py + ph/2

        tx1 = tx - tw/2
        ty1 = ty - th/2
        tx2 = tx + tw/2
        ty2 = ty + th/2

        # intersection coordinates
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        # intersection area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # union area
        pred_area = torch.clamp(pw, min=0) * torch.clamp(ph, min=0)
        target_area = torch.clamp(tw, min=0) * torch.clamp(th, min=0)
        union_area = pred_area + target_area - inter_area

        # IoU
        iou = inter_area / (union_area + self.eps)

        # IoU loss
        loss = 1 - iou

        # apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
