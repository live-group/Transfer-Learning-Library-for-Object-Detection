from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
# from US_DAF.LabelResizeLayer import ImageLabelResizeLayer
# from US_DAF.LabelResizeLayer import InstanceLabelResizeLayer



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,1,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        # self.LabelResizeLayer=ImageLabelResizeLayer()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        x = self.sigmoid(x)
        return x

# no gradient reverse layer
class class_ImageDA(nn.Module):
    def __init__(self,dim):
        super(class_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,1,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        x = self.sigmoid(x)
        return x

class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(2048, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,4)         # multi-label [domain, small, middle, large]
        # self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        # label = self.LabelResizeLayer(x, need_backprop)
        return x


