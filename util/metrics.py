from sklearn import metrics
from torch.nn import functional as F
import torch
import numpy as np


def mse_score(input, target):
    return F.mse_loss(input, target, reduction='elementwise_mean')

def f_score(y_true, y_pred, beta=1, threshold=0.5):
    return fbeta_score(y_true, y_pred, beta, threshold)

def paper_f1score(logits, targets, beta, eps=1e-9):
    beta = 1.0
    beta2 = beta ** 2.0
    logits = logits.float()
    targets = targets.float()
    top = torch.mul(targets, logits).sum()
    bot = beta2 * targets.sum() + logits.sum().add(eps)
    return torch.mean((1.0 + beta2) * top / bot)
    
def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum()
    precision = true_positive.div(y_pred.sum().add(eps))
    recall = true_positive.div(y_true.sum().add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

def f_score_no_threshold(y_true, y_pred, beta=1, threshold=None):
    return fbeta_score_no_threshold(y_true, y_pred, beta)

def fbeta_score_no_threshold(y_true, y_pred, beta, eps=1e-9):
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


if __name__ == "__main__":
    y_pred = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    y_true = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

    # y_pred = np.array([[[[1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.]]],


    #                 [[[1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.],
    #                     [1., 1., 1., 1., 1., 1., 1., 1.]]]])

    # y_true = np.array([[[[1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.],
    #                      [1., 1., 1., 1., 1., 1., 1., 1.]]],

    #                    [[[0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.],
    #                      [0., 0., 0., 0., 0., 0., 0., 0.]]]])         


    py_pred = torch.from_numpy(y_pred).view(-1)
    py_true = torch.from_numpy(y_true).view(-1)

    fbeta_pytorch = f_score(py_true, py_pred)
    fbeta_sklearn = metrics.fbeta_score(y_true, y_pred, 1, average='micro')

    # print('Scores are {:.5f} (sklearn) and {:.5f} (pytorch)'.format(fbeta_sklearn, fbeta_pytorch))
    # print("{:.5f}".format(paper_f1score(py_true, py_pred, 1)))