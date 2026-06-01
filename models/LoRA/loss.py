from pyparsing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lora_config import DEVICE

def keypoint_loss(
    pred_kps: torch.Tensor,
    gt_kps:   torch.Tensor,
    visibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """mse sur les keypoints pondéré par la visibilité. pred/gt : [B, N, 2]."""
    loss = F.mse_loss(pred_kps, gt_kps, reduction="none").sum(-1)  # [B, N]
    if visibility is not None:
        loss  = loss * (visibility > 0).float()
        denom = (visibility > 0).float().sum().clamp(min=1)
        return loss.sum() / denom
    return loss.mean()


def bbox_ciou_loss(
    pred_boxes: torch.Tensor,
    gt_boxes:   torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """ciou loss pour boîtes au format [cx, cy, w, h] normalisé."""

    def xywh2xyxy(b):
        return torch.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], dim=-1)

    pb = xywh2xyxy(pred_boxes)
    gb = xywh2xyxy(gt_boxes)

    inter_x1 = torch.max(pb[..., 0], gb[..., 0])
    inter_y1 = torch.max(pb[..., 1], gb[..., 1])
    inter_x2 = torch.min(pb[..., 2], gb[..., 2])
    inter_y2 = torch.min(pb[..., 3], gb[..., 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area_p = (pb[..., 2] - pb[..., 0]) * (pb[..., 3] - pb[..., 1])
    area_g = (gb[..., 2] - gb[..., 0]) * (gb[..., 3] - gb[..., 1])
    union  = area_p + area_g - inter + eps
    iou    = inter / union

    enc_x1 = torch.min(pb[..., 0], gb[..., 0])
    enc_y1 = torch.min(pb[..., 1], gb[..., 1])
    enc_x2 = torch.max(pb[..., 2], gb[..., 2])
    enc_y2 = torch.max(pb[..., 3], gb[..., 3])
    c2     = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    rho2 = ((pred_boxes[..., 0] - gt_boxes[..., 0]) ** 2 +
            (pred_boxes[..., 1] - gt_boxes[..., 1]) ** 2)

    with torch.no_grad():
        v = (4 / math.pi ** 2) * (
            torch.atan(gt_boxes[..., 2] / (gt_boxes[..., 3] + eps)) -
            torch.atan(pred_boxes[..., 2] / (pred_boxes[..., 3] + eps))
        ) ** 2
        alpha_c = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha_c * v
    return (1 - ciou).mean()


def lora_regularization(model: nn.Module, lambda_reg: float = 1e-4) -> torch.Tensor:
    """régularisation l2 légère sur les matrices lora entraînables."""
    reg = torch.tensor(0.0, device=DEVICE)
    for name, param in model.named_parameters():
        if param.requires_grad:
            reg = reg + param.norm(2) ** 2
    return lambda_reg * reg