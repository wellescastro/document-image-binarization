import torch
import torch.nn as nn
import torch.nn.functional as F

class F1ScoreLoss(nn.Module):
    def __init__(self):
        super(F1ScoreLoss, self).__init__()

    def forward(self, logits, targets):
        beta = 1.0
        beta2 = beta ** 2.0
        top = torch.mul(targets, logits).sum()
        bot = beta2 * targets.sum() + logits.sum()
        return 1-(1.0 + beta2) * top / bot
        
# def micro_fm(y_true, y_pred):
#     beta = 1.0
#     beta2 = beta**2.0
#     top = K.sum(y_true * y_pred)
#     bot = beta2 * K.sum(y_true) + K.sum(y_pred)
#     return -(1.0 + beta2) * top / bot


class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)