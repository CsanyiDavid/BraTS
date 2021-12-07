import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def dice_score(a, b):
    assert len(a.shape) == 5
    assert len(b.shape) == 5
    a = a>0.5
    b = b>0.5
    intersection = a * b
    axes = (1, 2, 3, 4)
    a_size = np.count_nonzero(a, axis = axes)
    b_size = np.count_nonzero(b, axis = axes)
    i_size = np.count_nonzero(intersection, axis = axes)
    dsc = (2 * i_size) / (a_size + b_size)
    mean_dsc = sum(dsc) / len(dsc)
    return mean_dsc

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, inputs, targets, smooth=0.0001):        
        #print(inputs.shape, targets.shape)
        #inputs = F.sigmoid(inputs)       
        
        axis = range(1, len(inputs.shape))
        axis = tuple(axis)
        #print('axis: ', axis)
        intersection = (inputs * targets).sum(axis=axis)
        #print('intersection: ', intersection)
        dice = (2.*intersection + smooth)/(inputs.sum(axis) + targets.sum(axis) + smooth)  
        #print('dice: ', dice)
        dice = dice.mean()
        #print('after mean: ', dice)
        return 1 - dice

class FocalTverskyLoss(nn.Module):
    def __init__(self):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = 0.7
        self.beta = 1- self.alpha
        self.gamma = 2
 
    def forward(self, inputs, targets, smooth=0.0001):        
        #print(inputs.shape, targets.shape)
        #inputs = F.sigmoid(inputs)       
        
        axis = range(1, len(inputs.shape))
        axis = tuple(axis)
        #print('axis: ', axis)
        tp = (inputs * targets).sum(axis=axis)
        fp = (inputs * (1-targets)).sum(axis=axis)
        fn = ((1-inputs) * targets).sum(axis=axis)

        tversky = (tp + smooth)/(tp + self.alpha*fn + self.beta*fp  + smooth)  
        focal_tversky = torch.pow(1-tversky, self.gamma)
        return focal_tversky.mean()

