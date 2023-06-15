import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from PT_MAF.rpn_cls import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

from PT_MAF.drm import DRM
from torch.autograd import Function
from PT_MAF.DA import _InstanceDA, _InstanceDA_w
from PT_MAF.LabelResizeLayer import ImageLabelResizeLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha=alpha* 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output, None

def grad_reverse(x, ratio):
    return GRLayer.apply(x, ratio)

class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop, ratio):
        x=grad_reverse(x, ratio)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
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

    def forward(self,x, need_backprop, ratio):
        x= grad_reverse(x, ratio)
        x= self.DRM(x)
        x= self.relu(self.Conv1(x))
        x= self.Conv2(x)

        label = self.LabelResizeLayer(x, need_backprop)

        return x, label

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA_3_f = _ImageDA_drm(256, 64, 4)
        self.RCNN_imageDA_4_f = _ImageDA_drm(512, 256, 2)
        self.RCNN_imageDA_f = _ImageDA(self.dout_base_model)

        self.RCNN_imageDA_3_b = _ImageDA_drm(256, 64, 4)
        self.RCNN_imageDA_4_b = _ImageDA_drm(512, 256, 2)
        self.RCNN_imageDA_b = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA_w(4105)

        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.num_anchor = len(self.anchor_scales)* len(self.anchor_ratios)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, T, high, low, alpha, beta, gamma):

        assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0

        batch_size = im_data.size(0)
        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop=need_backprop.data


        # feed image data to base model to obtain base feature map
        conv3_feat = self.conv3(im_data)
        conv4_feat = self.conv34(conv3_feat)
        base_feat = self.conv45(conv4_feat)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois, rpn_loss_cls, rpn_loss_bbox, rpn_score, rpn_prob = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rpn_prob_tmp, _ = torch.max(rpn_prob[:, self.num_anchor:, :, :], dim=1)
        max_prob = torch.max(rpn_prob_tmp)

        score_map_conv3_f = torch.where(rpn_prob_tmp.gt(max_prob* high), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
        score_map_conv3_b = torch.where(rpn_prob_tmp.lt(max_prob* low), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
#        score_map_conv3 = score_map_conv3_f + score_map_conv3_b

        score_map_conv4_f = torch.where(rpn_prob_tmp.gt(max_prob* high), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
        score_map_conv4_b = torch.where(rpn_prob_tmp.lt(max_prob* low), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
#        score_map_conv4 = score_map_conv4_1 + score_map_conv4_2

        score_map_conv5_f = torch.where(rpn_prob_tmp.gt(max_prob* high), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
        score_map_conv5_b = torch.where(rpn_prob_tmp.lt(max_prob* low), torch.full_like(rpn_prob_tmp, 1), torch.full_like(rpn_prob_tmp, 0))
#        score_map_conv5 = score_map_conv5_1 + score_map_conv5_2

        ratio_f = score_map_conv5_f.sum() / (score_map_conv5_b.sum()+score_map_conv5_f.sum()).detach()
        ratio_b = score_map_conv5_b.sum() / (score_map_conv5_b.sum()+score_map_conv5_f.sum()).detach()
        kd_rpn_prob = F.softmax(rpn_score / T, 1)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        kd_cls_prob = F.softmax(cls_score / T, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_conv3_feat = self.conv3(tgt_im_data)
        tgt_conv4_feat = self.conv34(tgt_conv3_feat)
        tgt_base_feat = self.conv45(tgt_conv4_feat)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox, _ ,tgt_rpn_prob = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)
        tgt_rpn_prob_tmp, _ = torch.max(tgt_rpn_prob[:, self.num_anchor:, :, :], dim=1)
        max_prob = torch.max(tgt_rpn_prob_tmp)

        tgt_score_map_conv3_f = torch.where(tgt_rpn_prob_tmp.gt(max_prob* high), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
        tgt_score_map_conv3_b = torch.where(tgt_rpn_prob_tmp.lt(max_prob* low), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
#        tgt_score_map_conv3 = tgt_score_map_conv3_1 + tgt_score_map_conv3_2

        tgt_score_map_conv4_f = torch.where(tgt_rpn_prob_tmp.gt(max_prob* high), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
        tgt_score_map_conv4_b = torch.where(tgt_rpn_prob_tmp.lt(max_prob* low), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
#        tgt_score_map_conv4 = tgt_score_map_conv4_f + tgt_score_map_conv4_b

        tgt_score_map_conv5_f = torch.where(tgt_rpn_prob_tmp.gt(max_prob* high), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
        tgt_score_map_conv5_b = torch.where(tgt_rpn_prob_tmp.lt(max_prob* low), torch.full_like(tgt_rpn_prob_tmp, 1), torch.full_like(tgt_rpn_prob_tmp, 0))
#        tgt_score_map_conv5 = tgt_score_map_conv5_1 + tgt_score_map_conv5_2

        tgt_ratio_f = tgt_score_map_conv5_f.sum() / (tgt_score_map_conv5_b.sum()+tgt_score_map_conv5_f.sum()).detach()
        tgt_ratio_b = tgt_score_map_conv5_b.sum() / (tgt_score_map_conv5_b.sum()+tgt_score_map_conv5_f.sum()).detach()
        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size)
            tgt_grid_yx = torch.stack([tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(tgt_base_feat, Variable(tgt_grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)
        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat)
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)


        """  DA loss   """

        # DA LOSS
        DA_img_loss_cls = 0
        DA_ins_loss_cls = 0

        tgt_DA_img_loss_cls = 0
        tgt_DA_ins_loss_cls = 0

        # DA conv3
        conv3_score_f, conv3_label_f = self.RCNN_imageDA_3_f(conv3_feat, need_backprop, ratio_f)

        conv3_prob_f = F.log_softmax(conv3_score_f, dim=1)
        conv3_label_f = torch.where(score_map_conv3_f == 1, conv3_label_f, torch.full_like(score_map_conv3_f, -1).long())
        DA_img_loss_conv3_f = F.nll_loss(conv3_prob_f, conv3_label_f, ignore_index=-1)

        conv3_score_b, conv3_label_b = self.RCNN_imageDA_3_b(conv3_feat, need_backprop, ratio_b)

        conv3_prob_b = F.log_softmax(conv3_score_f, dim=1)
        conv3_label_b = torch.where(score_map_conv3_b == 1, conv3_label_b, torch.full_like(score_map_conv3_b, -1).long())
        DA_img_loss_conv3_b = F.nll_loss(conv3_prob_b, conv3_label_b, ignore_index=-1)
        DA_img_loss_conv3 = DA_img_loss_conv3_b + DA_img_loss_conv3_f

        # DA conv4
        conv4_score_f, conv4_label_f = self.RCNN_imageDA_4_f(conv4_feat, need_backprop, ratio_f)

        conv4_prob_f = F.log_softmax(conv4_score_f, dim=1)
        conv4_label_f = torch.where(score_map_conv4_f == 1, conv4_label_f, torch.full_like(score_map_conv4_f, -1).long())
        DA_img_loss_conv4_f = F.nll_loss(conv4_prob_f, conv4_label_f, ignore_index=-1)

        conv4_score_b, conv4_label_b = self.RCNN_imageDA_4_b(conv4_feat, need_backprop, ratio_b)

        conv4_prob_b = F.log_softmax(conv4_score_b, dim=1)
        conv4_label_b = torch.where(score_map_conv4_b == 1, conv4_label_b, torch.full_like(score_map_conv4_b, -1).long())
        DA_img_loss_conv4_b = F.nll_loss(conv4_prob_b, conv4_label_b, ignore_index=-1)
        DA_img_loss_conv4 = DA_img_loss_conv4_b + DA_img_loss_conv4_f

        # DA conv5
        base_score_f, base_label_f = self.RCNN_imageDA_f(base_feat, need_backprop, ratio_f)
        base_prob_f = F.log_softmax(base_score_f, dim=1)
        base_label_f = torch.where(score_map_conv5_f == 1, base_label_f, torch.full_like(score_map_conv5_f, -1).long())
        DA_img_loss_conv5_f = F.nll_loss(base_prob_f, base_label_f, ignore_index=-1)

        base_score_b, base_label_b = self.RCNN_imageDA_b(base_feat, need_backprop, ratio_b)
        base_prob_b = F.log_softmax(base_score_b, dim=1)
        base_label_b = torch.where(score_map_conv5_b == 1, base_label_b, torch.full_like(score_map_conv5_b, -1).long())
        DA_img_loss_conv5_b = F.nll_loss(base_prob_b, base_label_b, ignore_index=-1)
        DA_img_loss_conv5 = DA_img_loss_conv5_f + DA_img_loss_conv5_b

        DA_img_loss_cls = alpha* DA_img_loss_conv3 + beta* DA_img_loss_conv4 + gamma* DA_img_loss_conv5

        # instance DA
        instance_feat = torch.cat((pooled_feat, cls_prob.squeeze()), dim=1)
#        print(instance_feat.size())
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(instance_feat, need_backprop)
        instance_loss = nn.CrossEntropyLoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        """  ************** target loss ****************  """

        # DA conv3 target
        tgt_conv3_score_f, tgt_conv3_label_f = \
            self.RCNN_imageDA_3_f(tgt_conv3_feat, tgt_need_backprop, tgt_ratio_f)

        tgt_conv3_label_f = torch.where(tgt_score_map_conv3_f == 1, tgt_conv3_label_f, torch.full_like(tgt_score_map_conv3_f, -1).long())
        tgt_conv3_prob_f = F.log_softmax(tgt_conv3_score_f, dim=1)
        tgt_DA_img_loss_conv3_f = F.nll_loss(tgt_conv3_prob_f, tgt_conv3_label_f, ignore_index=-1)

        tgt_conv3_score_b, tgt_conv3_label_b = \
            self.RCNN_imageDA_3_b(tgt_conv3_feat, tgt_need_backprop, tgt_ratio_b)

        tgt_conv3_label_b = torch.where(tgt_score_map_conv3_b == 1, tgt_conv3_label_b, torch.full_like(tgt_score_map_conv3_b, -1).long())
        tgt_conv3_prob_b = F.log_softmax(tgt_conv3_score_b, dim=1)
        tgt_DA_img_loss_conv3_b = F.nll_loss(tgt_conv3_prob_b, tgt_conv3_label_b, ignore_index=-1)
        tgt_DA_img_loss_conv3 = tgt_DA_img_loss_conv3_f + tgt_DA_img_loss_conv3_b

        # DA conv4 target
        tgt_conv4_score_f, tgt_conv4_label_f = \
            self.RCNN_imageDA_4_f(tgt_conv4_feat, tgt_need_backprop, tgt_ratio_f)

        tgt_conv4_prob_f = F.log_softmax(tgt_conv4_score_f, dim=1)
        tgt_conv4_label_f = torch.where(tgt_score_map_conv4_f == 1, tgt_conv4_label_f, torch.full_like(tgt_score_map_conv4_f, -1).long())
        tgt_DA_img_loss_conv4_f = F.nll_loss(tgt_conv4_prob_f, tgt_conv4_label_f, ignore_index=-1)

        tgt_conv4_score_b, tgt_conv4_label_b = \
            self.RCNN_imageDA_4_b(tgt_conv4_feat, tgt_need_backprop, tgt_ratio_b)

        tgt_conv4_prob_b = F.log_softmax(tgt_conv4_score_b, dim=1)
        tgt_conv4_label_b = torch.where(tgt_score_map_conv4_b == 1, tgt_conv4_label_b, torch.full_like(tgt_score_map_conv4_b, -1).long())
        tgt_DA_img_loss_conv4_b = F.nll_loss(tgt_conv4_prob_b, tgt_conv4_label_b, ignore_index=-1)
        tgt_DA_img_loss_conv4 = tgt_DA_img_loss_conv4_b + tgt_DA_img_loss_conv4_f

        # DA conv5 target
        tgt_base_score_f, tgt_base_label_f = \
            self.RCNN_imageDA_f(tgt_base_feat, tgt_need_backprop, tgt_ratio_f)

        tgt_base_prob_f = F.log_softmax(tgt_base_score_f, dim=1)
        tgt_base_label_f = torch.where(tgt_score_map_conv5_f == 1, tgt_base_label_f, torch.full_like(tgt_score_map_conv5_f, -1).long())
        tgt_DA_img_loss_conv5_f = F.nll_loss(tgt_base_prob_f, tgt_base_label_f, ignore_index=-1)

        tgt_base_score_b, tgt_base_label_b = \
            self.RCNN_imageDA_b(tgt_base_feat, tgt_need_backprop, tgt_ratio_b)

        tgt_base_prob_b = F.log_softmax(tgt_base_score_b, dim=1)
        tgt_base_label_b = torch.where(tgt_score_map_conv5_b == 1, tgt_base_label_b, torch.full_like(tgt_score_map_conv5_b, -1).long())
        tgt_DA_img_loss_conv5_b = F.nll_loss(tgt_base_prob_b, tgt_base_label_b, ignore_index=-1)
        tgt_DA_img_loss_conv5 = tgt_DA_img_loss_conv5_b + tgt_DA_img_loss_conv5_f

        '''
        print(score_map_conv3_f.sum(), tgt_score_map_conv3_f.sum())
        print(score_map_conv4_f.sum(), tgt_score_map_conv4_f.sum())
        print(score_map_conv5_f.sum(), tgt_score_map_conv5_f.sum())

        print(score_map_conv3_b.sum(), tgt_score_map_conv3_b.sum())
        print(score_map_conv4_b.sum(), tgt_score_map_conv4_b.sum())
        print(score_map_conv5_b.sum(), tgt_score_map_conv5_b.sum())
        '''

        tgt_DA_img_loss_cls = alpha* tgt_DA_img_loss_conv3 + beta* tgt_DA_img_loss_conv4 + gamma* tgt_DA_img_loss_conv5


        # instance_target
        tgt_instance_feat = torch.cat((tgt_pooled_feat, tgt_cls_prob.squeeze()), dim=1)
        tgt_instance_sigmoid, tgt_same_size_label = \
            self.RCNN_instanceDA(tgt_instance_feat, tgt_need_backprop)
        tgt_instance_loss = nn.CrossEntropyLoss()

        tgt_DA_ins_loss_cls = \
            tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
               DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls, kd_rpn_prob, kd_cls_prob


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()