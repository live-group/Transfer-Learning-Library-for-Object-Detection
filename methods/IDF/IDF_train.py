# coding:utf-8
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from IDF.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from IDF.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss


from IDF.parser_func import parse_args, set_dataset_args
# initilize the network here.
from IDF.vgg16 import vgg16


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.log_ckpt_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    num_boxes_p = torch.LongTensor(1)
    gt_boxes_p = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        num_boxes_p = num_boxes_p.cuda()
        gt_boxes_p = gt_boxes_p.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    num_boxes_p = Variable(num_boxes_p)
    gt_boxes_p = Variable(gt_boxes_p)
    if args.cuda:
        cfg.CUDA = True

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    #elif args.net == 'res101':
        #fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    # elif args.net == 'res50':
    #     fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    
    if args.cuda:
        fasterRCNN.cuda()
        
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


    if args.resume:
        load_name = os.path.join(output_dir,
           'DA_ObjectDetection_session_{}_epoch_{}_step_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        
    iters_per_epoch = int(10000 / args.batch_size)
    
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")
    
    txt_dist_dir = os.path.join(output_dir, 'record_dist.txt')
    txt_loss_dir = os.path.join(output_dir, 'record_loss.txt')
    
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        
        #isSeparation = False if epoch<0 else True
        isSeparation = False if epoch<3 else True
        #isSeparation = True
        count_step = 0
        loss_temp_last = 1  
        loss_temp = 0
        loss_temp2 = 0
        loss_rpn_cls_temp = 0
        loss_rpn_box_temp = 0
        loss_rcnn_cls_temp = 0
        loss_rcnn_box_temp = 0
        loss_rpn_cls_temp_t, loss_rpn_box_temp_t, loss_rcnn_cls_temp_t, loss_rcnn_box_temp_t = 0,0,0,0
        
        start = time.time()
        #if epoch % (args.lr_decay_step + 1) == 0:
        if epoch - 1 in  args.lr_decay_step:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)

        for step in range(1, iters_per_epoch + 1):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            #eta = 1.0

            #put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            if num_boxes > 20:
                print('num_boxes:{} is greater than 20'.format(num_boxes))


            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_1, out_d_2, out_d_3, out_d_ins, \
            priv_1, priv_2, priv_3, loss_s_se2, loss_s_se3, dist1_s, dist2_s, dist3_s  = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, gt_boxes, num_boxes, isSeparation)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            #out_d_ins_softmax = F.softmax(out_d_ins, 1) #[256,2]

            count_step += 1

            loss_temp += loss.item()
            loss_rpn_cls_temp += rpn_loss_cls.mean().item()
            loss_rpn_box_temp += rpn_loss_box.mean().item()
            loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
            loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()

            #################### source adversarial loss #############################
            # domain label
            domain_s_3 = Variable(torch.zeros(out_d_3.size(0)).long().cuda())
            # last featurn alignment loss
            dloss_s_3 = 0.5 * F.cross_entropy(out_d_3, domain_s_3)

            domain_s_2 = Variable(torch.zeros(out_d_2.size(0)).long().cuda())
            # mid feature alignment loss
            dloss_s_2 = 0.5 * F.cross_entropy(out_d_2, domain_s_2)
   
            domain_s_1 = Variable(torch.zeros(out_d_1.size(0)).long().cuda())
            #first feature alightment loss
            dloss_s_1 = 0.5 * F.cross_entropy(out_d_1, domain_s_1)

            # instance alignment loss
            domain_gt_ins = Variable(torch.zeros(out_d_ins.size(0)).long().cuda())
            dloss_s_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            #########################################################################
            
            ################### source non-adversarial loss##########################
            domain_s_3_b = Variable(torch.zeros(priv_3.size(0)).long().cuda())
            naloss_s_3 = 0.5 * F.cross_entropy(priv_3, domain_s_3_b)
            
            domain_s_2_b = Variable(torch.zeros(priv_2.size(0)).long().cuda())
            naloss_s_2 = 0.5 * F.cross_entropy(priv_2, domain_s_2_b)
            
            domain_s_1_b = Variable(torch.zeros(priv_1.size(0)).long().cuda())
            naloss_s_1 = 0.5 * F.cross_entropy(priv_1, domain_s_1_b)
            #########################################################################



            #put target data into variable
            im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()
            gt_boxes_p.data.resize_(data_t[2].size()).copy_(data_t[2])
            num_boxes_p.data.resize_(data_t[3].size()).copy_(data_t[3])

            rois_t, cls_prob_t, bbox_pred_t, \
            rpn_loss_cls_t, rpn_loss_box_t, \
            RCNN_loss_cls_t, RCNN_loss_bbox_t, \
            rois_label_t, out_d_1, out_d_2, out_d_3, out_d_ins, \
            priv_1, priv_2, priv_3, loss_t_se2, loss_t_se3, dist1_t, dist2_t, dist3_t  = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, gt_boxes_p, num_boxes_p, isSeparation, target=True)
            #out_d_ins_softmax = F.softmax(out_d_ins, 1)
            rpn_loss_cls_t *= 0.5
            rpn_loss_box_t *= 0.5
            RCNN_loss_cls_t *= 0.5
            RCNN_loss_bbox_t *= 0.5
            loss += rpn_loss_cls_t.mean() + rpn_loss_box_t.mean() \
                   + RCNN_loss_cls_t.mean() + RCNN_loss_bbox_t.mean()
            
            loss_temp2 += (rpn_loss_cls_t.mean() + rpn_loss_box_t.mean() \
                   + RCNN_loss_cls_t.mean() + RCNN_loss_bbox_t.mean()).item()
            loss_rpn_cls_temp_t += rpn_loss_cls_t.mean().item()
            loss_rpn_box_temp_t += rpn_loss_box_t.mean().item()
            loss_rcnn_cls_temp_t += RCNN_loss_cls_t.mean().item()
            loss_rcnn_box_temp_t += RCNN_loss_bbox_t.mean().item()
            
            #################### target adversarial loss #############################
            # domain label
            domain_t_3 = Variable(torch.ones(out_d_3.size(0)).long().cuda())
            dloss_t_3 = 0.5 * F.cross_entropy(out_d_3, domain_t_3)

            domain_t_2 = Variable(torch.ones(out_d_2.size(0)).long().cuda())
            dloss_t_2 = 0.5 * F.cross_entropy(out_d_2, domain_t_2)

            domain_t_1 = Variable(torch.ones(out_d_1.size(0)).long().cuda())
            dloss_t_1 = 0.5 * F.cross_entropy(out_d_1, domain_t_1)

            # instance alignment loss
            domain_gt_ins = Variable(torch.ones(out_d_ins.size(0)).long().cuda())
            dloss_t_ins = 0.5 * FL(out_d_ins, domain_gt_ins)
            ##########################################################################
            
            ################### target non-adversarial loss##########################
            domain_t_3_b = Variable(torch.ones(priv_3.size(0)).long().cuda())
            naloss_t_3 = 0.5 * F.cross_entropy(priv_3, domain_t_3_b)
            
            domain_t_2_b = Variable(torch.ones(priv_2.size(0)).long().cuda())
            naloss_t_2 = 0.5 * F.cross_entropy(priv_2, domain_t_2_b)
            
            domain_t_1_b = Variable(torch.ones(priv_1.size(0)).long().cuda())
            naloss_t_1 = 0.5 * F.cross_entropy(priv_1, domain_t_1_b)
            #########################################################################
            

            if isSeparation:
                loss += (dloss_s_3 + dloss_t_3 + dloss_s_1 + dloss_t_1 + dloss_s_2 + dloss_t_2 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5  \
                        + naloss_s_3 + naloss_s_2 + naloss_s_1 + naloss_t_3 + naloss_t_2 + naloss_t_1 + loss_s_se2 + loss_s_se3 + loss_t_se2 + loss_t_se3)    
            else:
                loss += (dloss_s_3 + dloss_t_3 + dloss_s_1 + dloss_t_1 + dloss_s_2 + dloss_t_2 + dloss_s_ins * 0.5 + dloss_t_ins * 0.5  \
                        + naloss_s_3 + naloss_s_2 + naloss_s_1 + naloss_t_3 + naloss_t_2 + naloss_t_1)
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()

                loss_temp /= count_step
                loss_temp2 /= count_step
                loss_rpn_cls_temp /= count_step
                loss_rpn_box_temp /= count_step
                loss_rcnn_cls_temp /= count_step
                loss_rcnn_box_temp /= count_step
                loss_rpn_cls_temp_t /= count_step
                loss_rpn_box_temp_t /= count_step
                loss_rcnn_cls_temp_t /= count_step
                loss_rcnn_box_temp_t /= count_step


                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s_3 = dloss_s_3.item()
                    dloss_t_3 = dloss_t_3.item()
                    dloss_s_1 = dloss_s_1.item()
                    dloss_t_1 = dloss_t_1.item()
                    dloss_s_2 = dloss_s_2.item()
                    dloss_t_2 = dloss_t_2.item()
                    dloss_s_ins = dloss_s_ins.item()
                    naloss_s_3 = naloss_s_3.item()
                    naloss_s_2 = naloss_s_2.item()
                    naloss_s_1 = naloss_s_1.item()
                    naloss_t_3 = naloss_t_3.item()
                    naloss_t_2 = naloss_t_2.item()
                    naloss_t_1 = naloss_t_1.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, loss2: %.4f, lr: %.2e, step: %3d" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, loss_temp2, lr, count_step))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f eta: %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, args.eta))
                print(
                    "\t\t\trpn_cls_t: %.4f, rpn_box_t: %.4f, rcnn_cls_t: %.4f, rcnn_box_t: %.4f" \
                    % (loss_rpn_cls_temp_t, loss_rpn_box_temp_t, loss_rcnn_cls_temp_t, loss_rcnn_box_temp_t))
                print(
                    "\t\t\tdloss s3: %.4f dloss t3: %.4f dloss s2: %.4f dloss t2: %.4f dloss s1: %.4f dloss t1: %.4f dloss_s_ins: %.4f dloss_t_ins: %.4f" \
                    % (dloss_s_3, dloss_t_3, dloss_s_2, dloss_t_2, dloss_s_1, dloss_t_1, dloss_s_ins, dloss_t_ins))
                print(
                    "\t\t\tnaloss s3: %.4f naloss t3: %.4f naloss s2: %.4f naloss t2: %.4f naloss s1: %.4f naloss t1: %.4f loss_s_se2: %.4f loss_s_se3: %.4f loss_t_se2: %.4f loss_t_se3: %.4f " \
                    % (naloss_s_3, naloss_t_3, naloss_s_2, naloss_t_2, naloss_s_1, naloss_t_1, loss_s_se2, loss_s_se3, loss_t_se2, loss_t_se3))
                txt_dist_file = open(txt_dist_dir, 'a')
                
                dist_result = ("[session %d][epoch %2d][iter %4d/%4d] dist1_s: %.4f, dist2_s: %.4f, dist3_s: %.4f, dist1_t: %.4f, dist2_t: %.4f, dist3_t: %.4f, " \
                           % (args.session, epoch, step, iters_per_epoch, dist1_s, dist2_s, dist3_s, dist1_t, dist2_t, dist3_t))
                dist_result = str(dist_result)
                txt_dist_file.write(dist_result + '\n')
                txt_dist_file.close()
                
                txt_loss_file = open(txt_loss_dir, 'a')
                loss_result = ("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, loss2: %.4f  \t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f eta: %.4f      \t\t\trpn_cls_t: %.4f, rpn_box_t: %.4f, rcnn_cls_t: %.4f, rcnn_box_t: %.4f    \t\t\tdloss s3: %.4f dloss t3: %.4f dloss s2: %.4f dloss t2: %.4f dloss s1: %.4f dloss t1: %.4f dloss_s_ins: %.4f dloss_t_ins: %.4f       \t\t\tnaloss s3: %.4f naloss t3: %.4f naloss s2: %.4f naloss t2: %.4f naloss s1: %.4f naloss t1: %.4f   \t\tloss_s_se2: %.4f loss_s_se3: %.4f loss_t_se2: %.4f loss_t_se3: %.4f " \
                           % (args.session, epoch, step, iters_per_epoch, loss_temp, loss_temp2, \
                            loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, args.eta, \
                            loss_rpn_cls_temp_t, loss_rpn_box_temp_t, loss_rcnn_cls_temp_t, loss_rcnn_box_temp_t,  \
                            dloss_s_3, dloss_t_3, dloss_s_2, dloss_t_2, dloss_s_1, dloss_t_1, dloss_s_ins, dloss_t_ins, \
                            naloss_s_3, naloss_t_3, naloss_s_2, naloss_t_2, naloss_s_1, naloss_t_1, loss_s_se2, loss_s_se3, loss_t_se2, loss_t_se3))
                loss_result = str(loss_result)
                txt_loss_file.write(loss_result + '\n')
                txt_loss_file.close()                
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls_temp,
                        'loss_rpn_box': loss_rpn_box_temp,
                        'loss_rcnn_cls': loss_rcnn_cls_temp,
                        'loss_rcnn_box': loss_rcnn_box_temp
                    }
                    # logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                    #                    (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalars(args.log_ckpt_name, info,
                                       (epoch - 1) * iters_per_epoch + step)

                count_step = 0
                loss_temp_last = loss_temp
                loss_temp = 0
                loss_temp2 = 0
                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0
                loss_rpn_cls_temp_t, loss_rpn_box_temp_t, loss_rcnn_cls_temp_t, loss_rcnn_box_temp_t = 0,0,0,0
                start = time.time()
            
            # if 6<epoch<12 and step%2000==0 and step!=10000:
            #     save_name = os.path.join(output_dir,
            #                      'DA_ObjectDetection_session_{}_epoch_{}_step_{}.pth'.format(
            #                          args.session, epoch,
            #                          step))
            #     save_checkpoint({
            #         'session': args.session,
            #         'epoch': epoch + 1,
            #         'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'pooling_mode': cfg.POOLING_MODE,
            #         'class_agnostic': args.class_agnostic,
            #     }, save_name)
            #     print('save model: {}'.format(save_name))
                
        save_name = os.path.join(output_dir,
                                 'DA_ObjectDetection_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
    

    if args.use_tfboard:
        logger.close()

