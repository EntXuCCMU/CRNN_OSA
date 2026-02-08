import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridEventLoss(nn.Module):
    """
    Combined loss function addressing class imbalance and event continuity.
    L_total = L_Focal + lambda * L_Dice
    """
    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.5, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, logits, targets, masks=None):
        # Flatten tensors
        logits_flat = logits.reshape(-1, self.num_classes)
        targets_flat = targets.reshape(-1)

        # Apply masking if regions of interest are defined
        if masks is not None:
            mask_flat = masks.reshape(-1).bool()
            if mask_flat.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]

        # 1. Focal Loss (Cross Entropy variant)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        focal_loss = focal_loss.mean()

        # 2. Dice Loss (Multiclass)
        probs = F.softmax(logits_flat, dim=-1)
        targets_one_hot = F.one_hot(targets_flat, num_classes=self.num_classes).float()

        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        eps = 1e-6
        dice_score = (2. * intersection + eps) / (union + eps)

        # Average Dice score over foreground classes (index 0 is 'Normal')
        event_dice = dice_score[1:].mean()
        dice_loss = 1.0 - event_dice

        return focal_loss + self.dice_weight * dice_loss