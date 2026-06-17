import torch
import torch.nn as nn

from Dice_loss import DiceLoss
from Boundary_loss import BoundaryLoss

class BCEDiceBoundaryLoss(nn.Module):
    def __init__(self, bce_weight=1.5, dice_weight=1.25, boundary_weight=0.75):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits, masks):
        bce_loss = self.bce(logits, masks)

        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, masks)
        boundary_loss = self.boundary(probs, masks)

        loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.boundary_weight * boundary_loss)

        return loss
