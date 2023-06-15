
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from IDF.faster_rcnn import _fasterRCNN

from model.utils.config import cfg
import os

import pdb
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "1x1 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

######first convolutional block
class netD_1(nn.Module):
    def __init__(self):
        super(netD_1, self).__init__()
        self.conv1 = conv1x1(256, 256, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256, 128, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 128, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x

class netD_1_b(nn.Module):
    def __init__(self):
        super(netD_1_b, self).__init__()
        self.conv1 = conv1x1(256, 256, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv1x1(256, 128, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv1x1(128, 128, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)    
        x = self.fc(x)
        return x

class fuse_add_1(nn.Module):
    def __init__(self):
        super(fuse_add_1, self).__init__()
        self.conv3 = conv3x3(256, 256, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = conv1x1(256, 256, stride=1)      
    def forward(self, x, y):
        x = torch.add(self.bn1(self.conv3(x)), self.bn1(self.conv3(y)))
        x = F.relu(self.conv1(x))
        return x

class fuse_concat_1(nn.Module):
    def __init__(self):
        super(fuse_concat_1, self).__init__()
        self.conv3 = conv3x3(256, 256, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = conv1x1(512, 256, stride=1)
    def forward(self, x, y):
        x = torch.cat((F.relu(self.bn1(self.conv3(x))), F.relu(self.bn1(self.conv3(y)))), 1)
        x = F.relu(self.conv1(x))
        return x

######second convolutional block
class netD_2(nn.Module):
    def __init__(self):
        super(netD_2, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x

class netD_2_b(nn.Module):
    def __init__(self):
        super(netD_2_b, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x

class fuse_add_2(nn.Module):
    def __init__(self):
        super(fuse_add_2, self).__init__()
        self.conv3 = conv3x3(512, 512, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1 = conv1x1(512, 512, stride=1)      
    def forward(self, x, y):
        x = torch.add(self.bn1(self.conv3(x)), self.bn1(self.conv3(y)))
        x = F.relu(self.conv1(x))
        return x
        
class fuse_concat_2(nn.Module):
    def __init__(self):
        super(fuse_concat_2, self).__init__()
        self.conv3 = conv3x3(512, 512, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1 = conv1x1(1024, 512, stride=1)
    def forward(self, x, y):
        x = torch.cat((F.relu(self.bn1(self.conv3(x))), F.relu(self.bn1(self.conv3(y)))), 1)
        x = F.relu(self.conv1(x))
        return x
        
######third convolutional block
class netD_3(nn.Module):
    def __init__(self):
        super(netD_3, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x

class netD_3_b(nn.Module):
    def __init__(self):
        super(netD_3_b, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        x = self.fc(x)
        return x

class fuse_add_3(nn.Module):
    def __init__(self):
        super(fuse_add_3, self).__init__()
        self.conv3 = conv3x3(512, 512, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1 = conv1x1(512, 512, stride=1)      
    def forward(self, x, y):
        x = torch.add(self.bn1(self.conv3(x)), self.bn1(self.conv3(y)))
        x = F.relu(self.conv1(x))
        return x

class fuse_concat_3(nn.Module):
    def __init__(self):
        super(fuse_concat_3, self).__init__()
        self.conv3 = conv3x3(512, 512, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1 = conv1x1(1024, 512, stride=1)
    def forward(self, x, y):
        x = torch.cat((F.relu(self.bn1(self.conv3(x))), F.relu(self.bn1(self.conv3(y)))), 1)
        x = F.relu(self.conv1(x))
        return x

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x


class netD_da(nn.Module):
    def __init__(self, feat_d):
        super(netD_da, self).__init__()
        self.fc1 = nn.Linear(feat_d,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x  #[256, 2]



class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    project_dir = os.path.join(os.path.relpath(__file__)[:os.path.relpath(__file__).rfind('/')],
                                 '../../')  # ../../lib/IDF/../../
    project_dir = os.path.abspath(project_dir)  # to show the path of file about pre-trained model explicitly

    self.model_path = os.path.join(project_dir, '../../pretrained_model/vgg16_caffe.pth')
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    #vgg_branch is target branch
    vgg_branch = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
        vgg_branch.load_state_dict({k:v for k,v in state_dict.items() if k in vgg_branch.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    vgg_branch.classifier = nn.Sequential(*list(vgg_branch.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    #print(vgg.features)
   
    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])
    self.netD_1 = netD_1()      #图像级域分类器
    self.netD_2 = netD_2()
    self.netD_3 = netD_3()
    feat_d = 4096
    self.netD_da = netD_da(feat_d)  #实例级域分类器
    
    #additional branch
    self.RCNN_base1_b = nn.Sequential(*list(vgg_branch.features._modules.values())[:14])
    self.RCNN_base2_b = nn.Sequential(*list(vgg_branch.features._modules.values())[14:21])
    self.RCNN_base3_b = nn.Sequential(*list(vgg_branch.features._modules.values())[21:-1])
    self.netD_1_b = netD_1_b()
    self.netD_2_b = netD_2_b()
    self.netD_3_b = netD_3_b()
    #self.fuse_1 = fuse_add_1()
    #self.fuse_2 = fuse_add_2()
    self.fuse_3 = fuse_add_3()
    #self.fuse_1 = fuse_concat_1()
    #self.fuse_2 = fuse_concat_2()
    #self.fuse_3 = fuse_concat_3()

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False
      for p in self.RCNN_base1_b[layer].parameters(): p.requires_grad = False


    self.RCNN_top = vgg.classifier
    self.RCNN_top_t = vgg_branch.classifier

    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
    
    #_t means additional network with target pseudo
    self.RCNN_cls_score_t = nn.Linear(4096, self.n_classes)
    if self.class_agnostic:
        self.RCNN_bbox_pred_t = nn.Linear(4096, 4)
    else:
        self.RCNN_bbox_pred_t = nn.Linear(4096, 4 * self.n_classes)
    

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)
    return fc7
    
  def _head_to_tail_t(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top_t(pool5_flat)
    return fc7
  


