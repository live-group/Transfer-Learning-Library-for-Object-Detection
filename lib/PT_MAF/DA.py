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

from PT_MAF.drm import DRM
from PT_MAF.LabelResizeLayer import ImageLabelResizeLayer
from PT_MAF.LabelResizeLayer import InstanceLabelResizeLayer



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

class WGRLayer(Function):

    @staticmethod
    def forward(ctx, input, score, dc_label):
        ctx.alpha = 0.2
        ctx.score = score
        ctx.dc_label = dc_label

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        score = ctx.score
        dc_label = ctx.dc_label.cpu().numpy()[0]

        weight = score[:, int(dc_label)].view(grad_output.shape[0], 1)
        weight = weight.repeat(1, grad_output.shape[1])
        output = grad_output.neg() *weight *ctx.alpha

        return output, None, None

def grad_reverse(x):
    return GRLayer.apply(x)

def wgrad_reverse(x, score, dc_label):
    return WGRLayer.apply(x, score, dc_label)

class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
        return x,label


class _InstanceDA_w(nn.Module):
    def __init__(self, input):
        super(_InstanceDA_w,self).__init__()
        self.dc_ip1 = nn.Linear(input, 1024)
        self.dc_relu1 = nn.ReLU()

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()

        self.clssifer=nn.Linear(1024,2)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x1 = torch.tensor(x)
        x1 = self.dc_relu1(self.dc_ip1(x1))
        x1 = self.dc_relu2(self.dc_ip2(x1))
        score = F.softmax(self.clssifer(x1), dim =1)

        x= wgrad_reverse(x, score, need_backprop)
        x= self.dc_relu1(self.dc_ip1(x))
        x= self.dc_relu2(self.dc_ip2(x))
#        x=F.sigmoid(self.clssifer(x))
        x = self.clssifer(x)
#        score1 = F.softmax(x, dim = 1)

        label = self.LabelResizeLayer(x, need_backprop)
        return x, label

class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4105, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x=F.sigmoid(self.clssifer(x))
        label = self.LabelResizeLayer(x, need_backprop)
        return x,label

class _ImageDA_drm(nn.Module):
    def __init__(self, dim, inner_channel, scale):
        super(_ImageDA_drm,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16

        self.DRM = DRM(self.dim, inner_channel, scale)
        self.DRM_out = inner_channel* scale* scale

        self.Conv1=nn.Conv2d(self.DRM_out,512,kernel_size=1,stride=1,bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x, need_backprop):
        x= grad_reverse(x)
        x= self.DRM(x)
        x= self.relu(self.Conv1(x))
        x= self.Conv2(x)

        label = self.LabelResizeLayer(x, need_backprop)

        return x, label

class _InstanceDA_cos(nn.Module):
    def __init__(self):
        super(_InstanceDA_cos,self).__init__()
        self.dc_ip1 = nn.Linear(4105, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.out_layer = nn.Linear(1024,2,bias=False)

        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x,need_backprop):
        F.normalize(self.out_layer.weight)

        x1 = torch.tensor(x)
        x1 = self.dc_relu1(self.dc_ip1(x1))
        x1 = self.dc_ip2(x1)
        x1 = F.normalize(x1)
        score=self.out_layer(x1)

        x=wgrad_reverse(x, score, need_backprop)
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_ip2(x)
        x=F.normalize(x)
#        ww=F.normalize(self.w, dim=0)
        out=self.out_layer(x)

        label = self.LabelResizeLayer(x, need_backprop)
        return out, label