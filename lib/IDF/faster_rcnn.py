import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

# from IDF.roi_align import ROIAlign
# from IDF.tooi_pool import ROIPool

from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from IDF.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, \
        _affine_theta, grad_reverse, avg_attention, cw_attention, dam

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

        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        
        ################# additional network just for target with pseudo #####################
        self.RCNN_loss_cls_t = 0
        self.RCNN_loss_bbox_t = 0
        self.RCNN_rpn_t = _RPN(self.dout_base_model)
        self.RCNN_proposal_target_t = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool_t = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align_t = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        ######################################################################################
        
    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_boxes_p, num_boxes_p, isSeparation=False, target=False, eta=1.0):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        
        # feed image data to base model to obtain base feature map
        #primary vgg16       
        base_feat1 = self.RCNN_base1(im_data)            
        domain_1 = self.netD_1(grad_reverse(base_feat1, lambd=eta))
        
        base_feat1_b = self.RCNN_base1_b(im_data)
        domain_1_b = self.netD_1_b(base_feat1_b)
        #
        dist1 = torch.mean(F.pairwise_distance(base_feat1, base_feat1_b, 2, keepdim=True))
        
        
        #4th block
        base_feat2 = self.RCNN_base2(base_feat1)
        domain_2 = self.netD_2(grad_reverse(base_feat2, lambd=eta))
        
        base_feat2_b = self.RCNN_base2_b(base_feat1_b)
        domain_2_b = self.netD_2_b(base_feat2_b)
        
        att2 = dam(base_feat2.detach())     # update domain-invariant branch    这里不应该detach 好像可以detach，这里就相当于一个权重
        att2_b = dam(base_feat2_b.detach())
        
        
        dist2 = torch.mean(F.pairwise_distance((base_feat2 * att2_b), (base_feat2_b * att2_b), 2, keepdim=True))
        loss_se_2 = 0.001*dist2

        base_feat2 = base_feat2 * (1 + att2_b)
        base_feat2_b = base_feat2_b * (1 + att2)


        #5th block    
        base_feat3 = self.RCNN_base3(base_feat2)
        domain_3 = self.netD_3(grad_reverse(base_feat3, lambd=eta))   
        
        base_feat3_b = self.RCNN_base3_b(base_feat2_b)
        domain_3_b = self.netD_3_b(base_feat3_b)

        att3 = dam(base_feat3.detach())
        att3_b = dam(base_feat3_b.detach())
        dist3 = torch.mean(F.pairwise_distance((base_feat3 * att3_b), (base_feat3_b * att3_b), 2, keepdim=True))
        loss_se_3 = 0.001*dist3        

        base_feat3 = base_feat3 * (1 + att3_b)
        base_feat3_b = base_feat3_b * (1 + att3)      
 
      
        #base_feat2_b = avg_attention(base_feat2_b, base_feat2_b.detach())        
        #base_feat3_b = avg_attention(base_feat3_b, base_feat3_b.detach())
        base_feat = base_feat3
 
        
        
        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground truth bboxes for refining
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

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # doamin adaptation on instance level 
        pooled_feat = self._head_to_tail(pooled_feat)
        domain_ins = self.netD_da(grad_reverse(pooled_feat, lambd=eta))
        
        
        ###################### additional network just for target with pseudo ##############################
        if target:
            ##########additional RPN network##########
            #print(gt_boxes_t)
            gt_boxes_p = gt_boxes_p.data
            num_boxes_p = num_boxes_p.data
            rois_t, rpn_loss_cls_t, rpn_loss_bbox_t = self.RCNN_rpn_t(base_feat3_b, im_info, gt_boxes_p, num_boxes_p)
            if self.training:
                roi_data_t = self.RCNN_proposal_target_t(rois_t, gt_boxes_p, num_boxes_p)
                rois_t, rois_label_t, rois_target_t, rois_inside_ws_t, rois_outside_ws_t = roi_data_t
                rois_label_t = Variable(rois_label_t.view(-1).long())
                rois_target_t = Variable(rois_target_t.view(-1, rois_target_t.size(2)))
                rois_inside_ws_t = Variable(rois_inside_ws_t.view(-1, rois_inside_ws_t.size(2)))
                rois_outside_ws_t = Variable(rois_outside_ws_t.view(-1, rois_outside_ws_t.size(2)))
            else:
                rois_label_t = None
                rois_target_t = None
                rois_inside_ws_t = None
                rois_outside_ws_t = None
                rpn_loss_cls_t = 0
                rpn_loss_bbox_t = 0
            
            rois_t = Variable(rois_t)
            if cfg.POOLING_MODE == 'align':
                pooled_feat_t = self.RCNN_roi_align_t(base_feat3_b, rois_t.view(-1, 5))
            elif cfg.POOLING_MODE == 'pool':
                pooled_feat_t = self.RCNN_roi_pool_t(base_feat3_b, rois_t.view(-1, 5))               
            pooled_feat_t = self._head_to_tail_t(pooled_feat_t)
            
            #########addition ROI_head###########
            bbox_pred_t = self.RCNN_bbox_pred_t(pooled_feat_t)
            if self.training and not self.class_agnostic:
                bbox_pred_view_t = bbox_pred_t.view(bbox_pred_t.size(0), int(bbox_pred_t.size(1) / 4), 4)
                bbox_pred_select_t = torch.gather(bbox_pred_view_t, 1, rois_label_t.view(rois_label_t.size(0), 1, 1).expand(rois_label_t.size(0), 1, 4))
                bbox_pred_t = bbox_pred_select_t.squeeze(1)
            cls_score_t = self.RCNN_cls_score_t(pooled_feat_t)
            cls_prob_t = F.softmax(cls_score_t, 1)
            
            RCNN_loss_cls_t = 0
            RCNN_loss_bbox_t = 0
            if self.training:
                RCNN_loss_cls_t = F.cross_entropy(cls_score_t, rois_label_t)
                RCNN_loss_bbox_t = _smooth_l1_loss(bbox_pred_t, rois_target_t, rois_inside_ws_t, rois_outside_ws_t)
                
            cls_prob_t = cls_prob_t.view(batch_size, rois_t.size(1), -1)
            bbox_pred_t = bbox_pred_t.view(batch_size, rois_t.size(1), -1)
            return rois_t, cls_prob_t, bbox_pred_t, rpn_loss_cls_t, rpn_loss_bbox_t, RCNN_loss_cls_t, RCNN_loss_bbox_t, rois_label_t, \
                       domain_1, domain_2, domain_3, domain_ins, \
                       domain_1_b, domain_2_b, domain_3_b, loss_se_2, loss_se_3 , dist1, dist2, dist3
        ##################################################################################################################################################




        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
                domain_1, domain_2, domain_3, domain_ins, \
                domain_1_b, domain_2_b, domain_3_b, loss_se_2, loss_se_3, dist1, dist2, dist3

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
        
        normal_init(self.RCNN_rpn_t.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_t.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn_t.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_t, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_t, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
