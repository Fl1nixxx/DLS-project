import torch 
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs, masks):
        masks = masks.float()

        intersection = (probs * masks).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()
