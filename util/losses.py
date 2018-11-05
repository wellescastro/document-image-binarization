import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from metrics import f_score, f_score_no_threshold

class F1ScoreLoss(nn.Module):
    '''
    dim=0 and logits_shape=(-1) calcula o fscore com base em um flat de todo o batch sem considerar o calculo por amostra
    dim=1 and logits_shape=(-1, 1, 256, 256) calcula o f1score pra cada patch e depois tira a media de scores
    dim=1 and logits_shape=(-1, 256 * 256) calcula o f1score pra cada flatted patch (parece a opcao mais razoavel)

    reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'


    '''
    def __init__(self):
        super(F1ScoreLoss, self).__init__()

    def forward(self, logits, targets, eps=1e-9):
        # logits_flat = logits.view(logits.shape[0], -1)
        # targets_flat = targets.view(targets.shape[0], -1)
        # logits = torch.ge(logits.float(), 0.5).float() # enable true f1-score 
        dim = 1
        beta = 1.0
        beta2 = beta ** 2.0
        top = torch.mul(targets, logits).sum(dim=0).add(eps)

        bot = beta2 * targets.sum(dim=0) + logits.sum(dim=0).add(eps)
        result = torch.mean(1 - ((1.0 + beta2) * top / bot))
        return result
        
# def micro_fm(y_true, y_pred):
#     beta = 1.0
#     beta2 = beta**2.0
#     top = K.sum(y_true * y_pred)
#     bot = beta2 * K.sum(y_true) + K.sum(y_pred)
#     return -(1.0 + beta2) * top / bot

class FBeta_ScoreLoss(nn.Module):

    def __init__(self, beta=1):
        super(FBeta_ScoreLoss, self).__init__()
        self.beta = beta
        self.threshold = 0.5
        # self.score_function = f_score_no_threshold

    def fbeta_score_no_threshold(self, y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2

        y_pred = y_pred.float()
        y_true = y_true.float()

        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))

        return torch.mean(
            (precision*recall).
            div(precision.mul(beta2) + recall + eps).
            mul(1 + beta2))

    def fbeta_score(self, y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2

        y_pred = torch.ge(y_pred.float(), self.threshold).float()
        y_true = y_true.float()

        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))

        return torch.mean(
            (precision*recall).
            div(precision.mul(beta2) + recall + eps).
            mul(1 + beta2))

    def forward(self, logits, targets):
        return (1 - self.fbeta_score(logits, targets, self.beta))


class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        Binary cross entropy loss 2D
        Args:
            weight:
            size_average:
        """
        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction='elementwise_mean')

    def forward(self, logits, targets):
        probs_flat = logits.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
