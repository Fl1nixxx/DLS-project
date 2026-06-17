import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, threshold=0.5, eps=1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_boundary(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

        boundary = torch.tanh(boundary * 4.0)
        return boundary

    def forward(self, probs, masks):
        masks = masks.float()

        pred_boundary = self.get_boundary(probs)
        true_boundary = self.get_boundary(masks)

        if self.threshold is not None:
            true_boundary = (true_boundary > self.threshold).float()

        intersection = (pred_boundary * true_boundary).sum(dim=(2, 3))
        union = pred_boundary.sum(dim=(2, 3)) + true_boundary.sum(dim=(2, 3))

        boundary_dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - boundary_dice.mean()

        return loss
