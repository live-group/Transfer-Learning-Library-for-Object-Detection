import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from PA_ATF.rpn2 import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from PA_ATF.LabelResizeLayer import ImageLabelResizeLayer
from PA_ATF.LabelResizeLayer import InstanceLabelResizeLayer
from torch.autograd import Function

# from model.da_faster_rcnn.DA import _ImageDA, _ImageDA_h, _ImageDA_drm
# from model.da_faster_rcnn.DA import Class_Base_Norm
# from model.da_faster_rcnn.DA import _InstanceDA
# from model.da_faster_rcnn.DA_new import _ImageDA, _ImageDAn_drm
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class GRLayer(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha

        return output, None

def grad_reverse(x, alpha):
    return GRLayer.apply(x, alpha)

class _InstanceDA(nn.Module):
    def __init__(self):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(4096, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=nn.Linear(1024,1)
        self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self, x, need_backprop, alpha=0.1):
        x = grad_reverse(x, alpha)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = F.sigmoid(self.clssifer(x))

        label = self.LabelResizeLayer(x, need_backprop)
        loss = torch.abs(x - label).mean()

        return loss

class _ImageDA(nn.Module):
    def __init__(self, dim):
        super(_ImageDA, self).__init__()
        self.dim=dim
        self.Conv1 = nn.Conv2d(self.dim, int(self.dim / 2), kernel_size=1, stride=1,bias=False)
        self.Conv2 = nn.Conv2d(int(self.dim / 2), 1, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)

        self.Convm1 = nn.Conv2d(self.dim, self.dim, kernel_size=5, stride=3, bias=True)
        self.reLum1 = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.Convm2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=2, bias=True)
        self.sig = nn.Sigmoid()

        self.LabelResizeLayer = ImageLabelResizeLayer()

    def forward(self, x, need_backprop, alpha=0.1):
        xx = grad_reverse(x, alpha)

        x_mask = self.reLum1(self.Convm1(xx))
        x_mask = self.Convm2(self.pool(x_mask))
        mask = self.sig(nn.functional.adaptive_max_pool2d(x_mask, (1, 1)))

        xx = xx* mask
        xx = self.reLu(self.Conv1(xx))
        xx = self.Conv2(xx)
        xx = F.sigmoid(xx)

        loss = nn.BCELoss()
        label = self.LabelResizeLayer(x, need_backprop).float()
#        score = F.log_softmax(xx, dim=1)
#        l1 = F.nll_loss(score, label, ignore_index=-1)
#        l1 = (xx - label).pow(2).mean()
        l1 = loss(xx, label)

        return l1, mask

class CLUB(nn.Module):
    def __init__(self, dim):
        super(CLUB, self).__init__()
        self.dim = dim

        self.out_score = nn.Sequential(nn.Conv2d(int(self.dim)* 2, self.dim, kernel_size=3, stride=2),\
                                       nn.ReLU(inplace=False), \
                                       nn.Conv2d(self.dim, 128, kernel_size=1, stride=1), \
                                       nn.ReLU(inplace=False))

        self.fc = nn.Linear(3*3*128, 2)

        '''
        self.Conv1 = nn.Conv2d(int(self.dim)* 2, self.dim, kernel_size=1, stride=1)
        self.reLu1 = nn.ReLU(inplace=False)
        self.Conv2 = nn.Conv2d(self.dim, 128, kernel_size=1, stride=1)

        self.fc = nn.linear()
        '''

    def forward(self, x1, x2, reverse):
        x11 = grad_reverse(x1, reverse)
        x22 = grad_reverse(x2, reverse)

        r_index = torch.randperm(x11.size(0))
        x22_r = x22[r_index, :, :, :]

        same_x = torch.cat((x11, x22), dim=1)
        diff_x = torch.cat((x11, x22_r), dim=1)

        same_conv = self.out_score(same_x)
        diff_conv = self.out_score(diff_x)

        same_score = self.fc(same_conv.view(same_conv.size(0), -1))
        diff_score = self.fc(diff_conv.view(diff_conv.size(0), -1))

        same_prob = F.log_softmax(same_score, dim=1)
        diff_prob = F.log_softmax(diff_score, dim=1)

        loss_same = F.nll_loss(same_prob, torch.ones(same_prob.size(0)).cuda().long())
        loss_diff = F.nll_loss(diff_prob, torch.zeros(diff_prob.size(0)).cuda().long())

        return loss_same + loss_diff

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

        self.RCNN_imageDA_3 = _ImageDA(256)
        self.RCNN_imageDA_4 = _ImageDA(512)
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()

        self.club3 = CLUB(256)
        self.club4 = CLUB(512)
        self.club5 = CLUB(self.dout_base_model)

        self.align1 = _RoIPooling(7, 7, 1.0/4.0)
        self.align2 = _RoIPooling(7, 7, 1.0/8.0)
        self.align3 = _RoIPooling(7, 7, 1.0/16.0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, weight = 0.1):

        assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0

        batch_size = im_data.size(0)
        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop=need_backprop.data


        # feed image data to base model to obtain base feature map
        conv3_feat = self.conv3_s(im_data)
        conv4_feat = self.conv34_s(conv3_feat)
        base_feat = self.conv45_s(conv4_feat)

        conv3_feat_t = self.conv3_t(im_data)
        conv4_feat_t = self.conv34_t(conv3_feat_t)
        base_feat_t = self.conv45_t(conv4_feat_t)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois_domain, rpn_loss_cls1, rpn_loss_bbox1, _ = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rois_domain_t, rpn_loss_cls2, rpn_loss_bbox2, _ = self.RCNN_rpn(base_feat_t, im_info, gt_boxes, num_boxes)
        rpn_loss_cls = rpn_loss_cls1 + rpn_loss_cls2
        rpn_loss_bbox = rpn_loss_bbox1 + rpn_loss_bbox2

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois_domain, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_data_t = self.RCNN_proposal_target(rois_domain_t, gt_boxes, num_boxes)
            rois_t, rois_label_t, rois_target_t, rois_inside_ws_t, rois_outside_ws_t = roi_data_t

            rois_label_t = Variable(rois_label_t.view(-1).long())
            rois_target_t = Variable(rois_target_t.view(-1, rois_target_t.size(2)))
            rois_inside_ws_t = Variable(rois_inside_ws_t.view(-1, rois_inside_ws_t.size(2)))
            rois_outside_ws_t = Variable(rois_outside_ws_t.view(-1, rois_outside_ws_t.size(2)))

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

#        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)

            grid_xy_t = _affine_grid_gen(rois_t.view(-1, 5), base_feat_t.size()[2:], self.grid_size)
            grid_yx_t = torch.stack([grid_xy_t.data[:,:,:,1], grid_xy_t.data[:,:,:,0]], 3).contiguous()
            pooled_feat_t = self.RCNN_roi_crop(base_feat_t, Variable(grid_yx_t).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_t = F.max_pool2d(pooled_feat_t, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_t = self.RCNN_roi_align(base_feat_t, rois_t.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            pooled_feat_t = self.RCNN_roi_pool(base_feat_t, rois_t.view(-1,5))

        # feed pooled features to top model
        pooled_feat_s = self._head_to_tail(pooled_feat)
        pooled_feat_t = self._head_to_tail(pooled_feat_t)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat_s)
        bbox_pred_t = self.RCNN_bbox_pred(pooled_feat_t)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            bbox_pred_view_t = bbox_pred_t.view(bbox_pred_t.size(0), int(bbox_pred_t.size(1) / 4), 4)
            bbox_pred_select_t = torch.gather(bbox_pred_view_t, 1, rois_label_t.view(rois_label_t.size(0), 1, 1).expand(rois_label_t.size(0), 1, 4))
            bbox_pred_t = bbox_pred_select_t.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat_s)
        cls_score_t = self.RCNN_cls_score(pooled_feat_t)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_s = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_cls_t = F.cross_entropy(cls_score_t, rois_label_t)
            RCNN_loss_cls = RCNN_loss_cls_s + RCNN_loss_cls_t

            # bounding box regression L1 loss
            RCNN_loss_bbox_s = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            RCNN_loss_bbox_t = _smooth_l1_loss(bbox_pred_t, rois_target_t, rois_inside_ws_t, rois_outside_ws_t)
            RCNN_loss_bbox = RCNN_loss_bbox_s + RCNN_loss_bbox_t

        cls_prob = cls_prob.view(batch_size, rois_t.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois_t.size(1), -1)

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1, size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_conv3_feat = self.conv3_s(tgt_im_data)
        tgt_conv4_feat = self.conv34_s(tgt_conv3_feat)
        tgt_base_feat = self.conv45_s(tgt_conv4_feat)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        cfg.TEST.RPN_POST_NMS_TOP_N = rois.size(1)
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox, tgt_anchor_score = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

        tgt_rois = Variable(tgt_rois)

        if cfg.POOLING_MODE == 'crop':
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

        """  DA loss   """

        # DA conv3
        DA_img_loss_conv3, cw3 = self.RCNN_imageDA_3(conv3_feat_t, need_backprop, weight)

        # DA conv4
        DA_img_loss_conv4, cw4 = self.RCNN_imageDA_4(conv4_feat_t, need_backprop, weight)

        # DA conv5
        DA_img_loss_conv5, cw5 = self.RCNN_imageDA(base_feat_t, need_backprop, weight)
        DA_img_loss_cls = DA_img_loss_conv3 + DA_img_loss_conv4 + DA_img_loss_conv5
#        DA_img_loss_cls = DA_img_loss_conv5

        DA_ins_loss_cls = self.RCNN_instanceDA(pooled_feat_t, need_backprop)

        '''
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(pooled_feat_t, need_backprop)
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)
        '''


        """  ************** target loss ****************  """

        # DA conv3 target
        tgt_DA_img_loss_conv3, cw3_t = self.RCNN_imageDA_3(tgt_conv3_feat, tgt_need_backprop, weight)

        # DA conv4 target
        tgt_DA_img_loss_conv4, cw4_t = self.RCNN_imageDA_4(tgt_conv4_feat, tgt_need_backprop, weight)

        # DA conv5 target
        tgt_DA_img_loss_conv5, cw5_t = self.RCNN_imageDA(tgt_base_feat, tgt_need_backprop, weight)
        tgt_DA_img_loss_cls = tgt_DA_img_loss_conv3 + tgt_DA_img_loss_conv4 + tgt_DA_img_loss_conv5

        # instance_target
        tgt_DA_ins_loss_cls = self.RCNN_instanceDA(tgt_pooled_feat, tgt_need_backprop)

#        tgt_instance_sigmoid, tgt_same_size_label = self.RCNN_instanceDA(tgt_pooled_feat, tgt_need_backprop)
#        tgt_instance_loss = nn.BCELoss()
#        tgt_DA_ins_loss_cls = tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

#        cw3 = self.RCNN_imageDA_3.mask(conv3_feat_t, False)
#        cw4 = self.RCNN_imageDA_4.mask(conv4_feat_t, False)
#        cw5 = self.RCNN_imageDA.mask(base_feat_t, False)

        '''
        conv3_feat_align = conv3_feat_t* cw3.detach()
        conv4_feat_align = conv4_feat_t* cw4.detach()
        base_feat_align = base_feat_t* cw5.detach()

        conv3_feat_spc = conv3_feat_t* (1. - cw3.detach())
        conv4_feat_spc = conv4_feat_t* (1. - cw4.detach())
        base_feat_spc = base_feat_t* (1. - cw5.detach())
        '''

        gt_rois = gt_boxes.squeeze()[:num_boxes, :-1]
        gt_rois = torch.cat((torch.zeros(gt_rois.size(0), 1).cuda(), gt_rois), dim=1)
        gt_rois = Variable(gt_rois)

        roi_conv3 = self.align1(conv3_feat_t, gt_rois)
        roi_conv4 = self.align2(conv4_feat_t, gt_rois)
        roi_conv5 = self.align3(base_feat_t, gt_rois)

        f_conv3_a = roi_conv3* cw3.detach()
        f_conv4_a = roi_conv4* cw4.detach()
        f_conv5_a = roi_conv5* cw5.detach()

        f_conv3_s = roi_conv3* (1 - cw3.detach())
        f_conv4_s = roi_conv4* (1 - cw4.detach())
        f_conv5_s = roi_conv5* (1 - cw5.detach())

        pm3 = self.club3(f_conv3_a, f_conv3_s, 0.1)
        pm4 = self.club4(f_conv4_a, f_conv4_s, 0.1)
        pm5 = self.club5(f_conv5_a, f_conv5_s, 0.1)
        pm_loss = pm3 + pm4 + pm5

        return rois_t, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label_t, \
               DA_img_loss_cls, tgt_DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_ins_loss_cls, pm_loss


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
