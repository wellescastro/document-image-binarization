import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .metrics import f_score, f_score_no_threshold

class F1ScoreLoss(nn.Module):
    '''
    dim=0 and logits_shape=(-1) calcula o fscore com base em um flat de todo o batch sem considerar o calculo por amostra
    dim=1 and logits_shape=(-1, 1, 256, 256) calcula o f1score pra cada patch e depois tira a media de scores
    dim=1 and logits_shape=(-1, 256 * 256) calcula o f1score pra cada flatted patch (parece a opcao mais razoavel)
    dim=None and logits_shape=(-1, 256 * 256) calcula o micro f1 (across the classes instead of f1 for each class )

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
        beta = 1.0
        beta2 = beta ** 2.0
        top = torch.mul(targets, logits).sum()
        bot = beta2 * targets.sum() + logits.sum().add(eps)
        f1 = ((1.0 + beta2) * top / bot)
        result = torch.mean(1 - ((1.0 + beta2) * top / bot))
        return result
        

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

        true_positive = (y_pred * y_true).sum()
        precision = true_positive.div(y_pred.sum().add(eps))
        recall = true_positive.div(y_true.sum().add(eps))

        return (precision*recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2)

    def fbeta_score(self, y_true, y_pred, beta, eps=1e-9):
        beta2 = beta**2

        y_pred = torch.ge(y_pred.float(), self.threshold).float()
        y_true = y_true.float()

        true_positive = (y_pred * y_true).sum()
        precision = true_positive.div(y_pred.sum().add(eps))
        recall = true_positive.div(y_true.sum().add(eps))

        return torch.mean(
            (precision*recall).
            div(precision.mul(beta2) + recall + eps).
            mul(1 + beta2))

    def forward(self, logits, targets):
        return (1 - self.fbeta_score_no_threshold(logits, targets, self.beta))
