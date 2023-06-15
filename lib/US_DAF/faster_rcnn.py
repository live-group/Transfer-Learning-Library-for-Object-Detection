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
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import math
from US_DAF.DA import _ImageDA,class_ImageDA
from US_DAF.DA import _InstanceDA
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

# from model.da_faster_rcnn.UDA import get_source_share_weight,get_target_share_weight,normalize_weight


def BCEloss_margin(x_sigmoid, label):
    NEAR_0 = 1e-10
    scale_weight=torch.ones(len(label),3).cuda()
    BCEloss = -(label * torch.log(x_sigmoid+NEAR_0) + (1 - label) * torch.log(1 - x_sigmoid+NEAR_0))
    n = len(BCEloss)#数据个数
    a = (BCEloss[:, 0] > 0.5).reshape(n, -1).float()#将第一列满足条件的为1,否则为0（第一列代表是否为公共类）
    weight=torch.cat([a,scale_weight],dim=1)
    BCEloss = BCEloss * weight
    return BCEloss.mean()


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

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_imageDA_class = class_ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)

    def forward(self, im_data, im_info, gt_boxes, num_boxes,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes):
        need_backprop=torch.tensor([1.0]).cuda()
        tgt_need_backprop = torch.tensor([0.0]).cuda()


        assert need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0

        batch_size = im_data.size(0)
        im_info = im_info.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data
        #print("输入数据的大小",im_data.shape)
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)  # 得到的base_feat将分为两路，一路进行rois，一路原封不动，直接将与得到的rois的进行roi pooling。
        # feed base feature map tp RPN to obtain rois
        # print("输出数据的大小", base_feat.shape)
        self.RCNN_rpn.train()
        # print("我就想知道这句是啥意思",self.RCNN_rpn.train())
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes,num_boxes)  # base_feat输入，输出rois，针对源域数据这里大约2000个
        # if it is training phrase, then use ground trubut bboxes for refining

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data  # 这里的rois有256个

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

        rois = Variable(rois)  # [1, 256, 5]

        # prepare the scale label for rois
        roi_small = []
        roi_middle = []
        roi_large = []
        rois2 = rois.squeeze(0)
        for i in range(len(rois2)):
            x1 = rois2[i][1]
            y1 = rois2[i][2]
            x2 = rois2[i][3]
            y2 = rois2[i][4]
            area = (x2 - x1) * (y2 - y1)
            if area <= 400:
                roi_small.append(i)
            elif (area > 400) & (area < 10000):
                roi_middle.append(i)
            elif area >= 10000:
                roi_large.append(i)

        source_small = torch.zeros(len(rois2), 1)
        source_small[roi_small] = 1
        source_middle = torch.zeros(len(rois2), 1)
        source_middle[roi_middle] = 1
        source_large = torch.zeros(len(rois2), 1)
        source_large[roi_large] = 1

        if cfg.POOLING_MODE == 'align':# 默认选的是这个
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))  # torch.Size([256, 512, 7, 7])


        pooled_feat = self._head_to_tail(pooled_feat)  # vgg16是torch.Size([256, 4096])#res101是[128,2048]
        # print("输出pooled_feat的维度",pooled_feat.shape)

        # compute bbox offset,roi池化后提取的roi特征计算边框预测值
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)  # 用于回归
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)  # 用于分类#[128,21]
        cls_prob2 = F.softmax(cls_score, 1)#[128,21]
        # print("源域的输出维度",  cls_prob2.shape)
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        # 图像级别的权重
        _, _, h1, w1 = im_data.size()
        _, _, h2, w2 = base_feat.size()
        # print("特征图的大小",base_feat.shape)
        label_img = torch.zeros(size=(self.n_classes,h2, w2)).cuda()  ##################这里不同于1_1,将非目标区域标记为0
        # print("权重的整体大小", weight_ad_img.shape)
        scale_w = w2 / w1
        scale_h = h2 / h1

        gt_rois = rois.new(rois.size()).zero_()
        gt_rois[:, :, 1:5] = rois[:, :, 1:5]
        gt_rois[:, :, 0] = rois_label

        gt_boxes2 = gt_rois.squeeze(0)



        for i in range(len(gt_boxes2)):
                x1 = torch.tensor(math.floor(gt_boxes2[i][1] * scale_w))
                y1 = torch.tensor(math.floor(gt_boxes2[i][2] * scale_h))
                x2 = torch.tensor(math.floor(gt_boxes2[i][3] * scale_w))
                y2 = torch.tensor(math.floor(gt_boxes2[i][4] * scale_h))
                weight_gt = torch.ones((self.n_classes, y2 - y1 + 1, x2 - x1 + 1)).cuda() * cls_prob2[i][:].view( self.n_classes, 1, 1)

                label_img[:, y1:y2 + 1, x1:x2 + 1] = weight_gt

        image_softmax = label_img.view(self.n_classes, -1).t()

        if self.training:  # 对源域数据进行分类weight_s2.detach()
            # img_loss_cls = F.cross_entropy(cls_score_img2, label_img.view(-1).long())
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob2.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data


        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.eval()
        # print("我就想知道这第二句是啥意思", self.RCNN_rpn.eval())
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)
        tgt_rois = Variable(tgt_rois)

        tgt_roi_small = []
        tgt_roi_middle = []
        tgt_roi_large = []
        tgt_rois2 = tgt_rois.squeeze(0)

        for i in range(len(tgt_rois2)):
            tx1 = tgt_rois2[i][1]
            ty1 = tgt_rois2[i][2]
            tx2 = tgt_rois2[i][3]
            ty2 = tgt_rois2[i][4]
            tarea = (tx2 - tx1) * (ty2 - ty1)
            if tarea <= 400:
                tgt_roi_small.append(i)
            elif (tarea > 400) & (tarea < 10000):
                tgt_roi_middle.append(i)
            elif tarea >= 10000:
                tgt_roi_large.append(i)

        # do roi pooling based on predicted rois
        target_small = torch.zeros(len(tgt_rois2), 1)
        target_small[tgt_roi_small] = 1
        target_middle = torch.zeros(len(tgt_rois2), 1)
        target_middle[tgt_roi_middle] = 1
        target_large = torch.zeros(len(tgt_rois2), 1)
        target_large[tgt_roi_large] = 1

        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))  # torch.Size([300, 512, 7, 7])
        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat)  # 用于分类#[300,21]
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)  # [300,21]

        # # 图像级别的权重
        _, _, th1, tw1 = tgt_im_data.size()
        _, _, th2, tw2 = tgt_base_feat.size()
        # print("特征图的大小",base_feat.shape)
        tgt_label_img = torch.zeros(size=(self.n_classes,th2, tw2)).cuda()  ##################这里不同于1_1,将非目标区域标记为0
        # print("target权重的整体大小", tgt_weight_ad_img.shape)
        t_scale_w = tw2 / tw1
        t_scale_h = th2 / th1
        gt_tgt_rois2 = tgt_rois.squeeze(0)

        for i in range(len(gt_tgt_rois2)):

                    t_x1 = torch.tensor(math.floor(gt_tgt_rois2[i][1] * t_scale_w))
                    t_y1 = torch.tensor(math.floor(gt_tgt_rois2[i][2] * t_scale_h))
                    t_x2 = torch.tensor(math.floor(gt_tgt_rois2[i][3] * t_scale_w))
                    t_y2 = torch.tensor(math.floor(gt_tgt_rois2[i][4] * t_scale_h))
                    tgt_weight_gt = torch.ones((self.n_classes,t_y2 - t_y1 + 1, t_x2 - t_x1 + 1)).cuda() * tgt_cls_prob[i][:].view( self.n_classes, 1, 1)
                    # print("真实gt的权重大小", weight_gt.shape)

                    tgt_label_img[:, t_y1:t_y2 + 1, t_x1:t_x2 + 1] = tgt_weight_gt
        tgt_image_softmax = tgt_label_img.view(self.n_classes, -1).t()

        "源域对齐"
        base_score = self.RCNN_imageDA(base_feat)
        DA_img_loss_cls = nn.BCELoss()(base_score.view(-1, 1), torch.ones_like(base_score).view(-1, 1))

        instance_sigmoid = self.RCNN_instanceDA(pooled_feat)
        source_domain_label = torch.ones(len(rois2), 1)
        source_label_ins = torch.cat([source_domain_label, source_small, source_middle, source_large], dim=1).cuda()    # multi-label
        DA_ins_loss_cls = BCEloss_margin(instance_sigmoid, source_label_ins)
        # print("源域实例级损失",DA_ins_loss_cls)

        """  ************** taget loss ****************  """

        tgt_base_score = self.RCNN_imageDA(tgt_base_feat)
        tgt_DA_img_loss_cls = nn.BCELoss()(tgt_base_score.view(-1, 1), torch.zeros_like(tgt_base_score).view(-1, 1))

        tgt_instance_sigmoid = self.RCNN_instanceDA(tgt_pooled_feat)
        target_domain_label = torch.zeros(len(tgt_rois2), 1)
        target_label_ins = torch.cat([target_domain_label, target_small, target_middle, target_large], dim=1).cuda()    # multi-label
        tgt_DA_ins_loss_cls = BCEloss_margin(tgt_instance_sigmoid, target_label_ins)
        # print("目标域实例级损失", tgt_DA_ins_loss_cls)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,rois_label, \
               DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_cls_score_img, 0, 0.01, cfg.TRAIN.TRUNCATED)
    def create_architecture(self):
        self._init_modules()
        self._init_weights()
