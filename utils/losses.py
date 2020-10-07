#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable, Function
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes ### shape[1]
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


# def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
#     # assumes that input is a normalized probability
#     assert input.size == target.size, "'input' and 'target' must have hte same size"
#     if ignore_index is not None:
#         mask = target.clone().ne_(ignore_index)
#         mask.requires_grad = False
#         input = input * mask
#         target = target * mask
#
#     input = flatten(input)
#     target = flatten(target)
#
#     target = target.float()
#     # compute per channel dice
#     intersect = (input * target).sum(-1)
#     if weight is not None:
#         intersect = weight * intersect
#
#     denominator = (input + target).sum(-1)
#     return 2.0 * intersect / denominator.clamp(min=epsilon)
#
#
# class DiceLoss(nn.Module):
#     def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False,
#                  skip_last_target=False):
#         super(DiceLoss, self).__init__()
#         self.epsilon = epsilon
#         self.register_buffer('weight', weight)
#         self.ignore_index = ignore_index
#         if sigmoid_normalization:
#             self.normalization = nn.Sigmoid()
#         else:
#             self.normalization = nn.Softmax(dim=1)
#
#         self.skip_last_target = skip_last_target
#
#     def forward(self, input, target):
#         input = self.normalization
#         if self.weight is not None:
#             weight = Variable(self.weight, requires_grad=False)
#         else:
#             weight = None
#
#         if self.skip_last_target:
#             target = target[:, :-1, ...]
#
#         per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
#                                                     weight=weight)
#         return torch.mean(1.0 - per_channel_dice)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # target = torch.LongTensor(target)
        # target = make_one_hot(target, num_classes=predict.shape[1])   ###   num_classes=predict.shape[1]
        target = target.cuda()
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


# ---------------------------------------------------------------------------------------------------------
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        #target = _make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 0.0001
        #dot是返回两个矩阵的点集
        #inter,uniun:两个值的大小分别是10506.6,164867.2
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        #print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None


        #这里没有打印出来，难道没有执行到这里吗
        #print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    #print("size of input, target:", input.shape, target.shape)

    for i, c in enumerate(zip(input, target)):
        #c[0],c[1]的大小都是原图大小torch.Size([1, 576, 544])
        #print("size of c0 c1:", c[0].shape,c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def dice_coeff_loss(input, target):

    return 1 - dice_coeff(input, target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


# ---------------------------------------------------------------------------------------------

def log_sum_exp(x):
    # b is a shift factor to avoid overflow
    # x.size() = [N,C]
    b, _ = torch.max(x, 1)
    y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
    return y.squeeze(1)


def class_select(logits, target):
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(
            torch.arange(0, num_classes).long().repeat(batch_size, 1).cuda(device).eq(
                target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(
            torch.arange(0, num_classes).long().repeat(batch_size, 1).eq(target.data.repeat(num_classes, 1).t()))

    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        assert list(loss.size()) == list(weights.size())
        loss = loss * weights
    return loss


class WeightCELoss(nn.Module):
    def __init__(self, aggregate='mean'):
        super(WeightCELoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)

# ---------------------------------------------------------------------------------------------------
