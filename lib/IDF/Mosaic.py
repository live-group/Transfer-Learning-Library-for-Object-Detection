# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
import random
import sys

import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

import xml.etree.ElementTree as ET

def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


def fill_truth_detection(bboxes, num_boxes, classes, flip, dx, dy, sx, sy, net_w, net_h):
    if bboxes.shape[0] == 0:
        return bboxes, 10000
    np.random.shuffle(bboxes)
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    if bboxes.shape[0] == 0:
        return bboxes, 10000

    bboxes = bboxes[np.where((bboxes[:, 4] < classes) & (bboxes[:, 4] >= 0))[0]]

    if bboxes.shape[0] > num_boxes:
        bboxes = bboxes[:num_boxes]

    min_w_h = np.array([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).min()

    bboxes[:, 0] *= (net_w / sx)
    bboxes[:, 2] *= (net_w / sx)
    bboxes[:, 1] *= (net_h / sy)
    bboxes[:, 3] *= (net_h / sy)

    if flip:
        temp = net_w - bboxes[:, 0]
        bboxes[:, 0] = net_w - bboxes[:, 2]
        bboxes[:, 2] = temp

    return bboxes, min_w_h


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            truth):
    try:
        img = mat
        oh, ow, _ = img.shape
        pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
        # crop
        src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        new_src_rect = rect_intersection(src_rect, img_rect)  # 交集

        dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
        # cv2.Mat sized

        if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
            sized = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        else:
            cropped = np.zeros([sheight, swidth, 3])
            cropped[:, :, ] = np.mean(img, axis=(0, 1))

            cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
                img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

            # resize
            sized = cv2.resize(cropped, (w, h), cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            sized = cv2.flip(sized, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(sized.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
                hsv = cv2.split(hsv_src)
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                sized = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        if blur:
            if blur == 1:
                dst = cv2.GaussianBlur(sized, (17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                dst = cv2.GaussianBlur(sized, (ksize, ksize), 0)

            if blur == 1:
                img_rect = [0, 0, sized.cols, sized.rows]
                for b in truth:
                    left = (b.x - b.w / 2.) * sized.shape[1]
                    width = b.w * sized.shape[1]
                    top = (b.y - b.h / 2.) * sized.shape[0]
                    height = b.h * sized.shape[0]
                    roi(left, top, width, height)
                    roi = roi & img_rect
                    dst[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]] = sized[roi[0]:roi[0] + roi[2],
                                                                          roi[1]:roi[1] + roi[3]]

            sized = dst

        if gaussian_noise:
            noise = np.array(sized.shape)
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized = sized + noise
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        sized = mat

    return sized


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    if bboxes.size == 0:
        return np.empty((0, 5), dtype='float64')
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                       left_shift, right_shift, top_shift, bot_shift):
    left_shift = min(left_shift, w - cut_x)
    top_shift = min(top_shift, h - cut_y)
    right_shift = min(right_shift, cut_x)
    bot_shift = min(bot_shift, cut_y)

    if i_mixup == 0:
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if i_mixup == 1:
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
    if i_mixup == 2:
        bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
    if i_mixup == 3:
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

    return out_img, bboxes


def draw_box(img, bboxes):
    for b in bboxes:
        img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
    return img

def _load_pascal_annotation(data_path, index, class_to_ind):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(data_path, 'VOC2007/Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = max(float(bbox.find('xmin').text) - 1, 0)
            y1 = max(float(bbox.find('ymin').text) - 1, 0)
            x2 = max(float(bbox.find('xmax').text) - 1, 0)
            y2 = max(float(bbox.find('ymax').text) - 1, 0)

            cls = class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]

            gt_classes[ix] = cls


        return {'boxes': boxes,
                'gt_classes': gt_classes
                }


class Yolo_dataset(Dataset):
    def __init__(self, label_path, cfg, train=True):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg
        self.train = train

        self._classes = ('__background__',  # always index 0
                         'bus', 'bicycle', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
        self._class_to_ind = dict(zip(self._classes, range(len(self._classes))))

        truth = {}  # {image_name(key):bounding boxed(value)}
        f = open(label_path, 'r', encoding='utf-8')
        for line in f.readlines():
            # data = line.split(" ")
            data = line.strip().split(" ")

            truth[data[0]] = []
            # 添加bbox到value, load bbox from VOC
            annotation = _load_pascal_annotation(self.cfg.dataset_dir, data[0], self._class_to_ind)
            annotation = np.concatenate( (annotation['boxes'], annotation['gt_classes'].reshape(-1, 1) ), axis=1)
            for i in range(len(annotation)):
                truth[data[0]].append(annotation[i])


            # for i in data[1:]:  # 添加bbox到value
            #     truth[data[0]].append([int(float(j)) for j in i.split(',')])

        self.truth = truth
        self.imgs = list(self.truth.keys())


    def __len__(self):
        return len(self.truth.keys())

    def __getitem__(self, index):
        if not self.train:
            return self._get_val_item(index)
        img_name = self.imgs[index]

        # annotation = _load_pascal_annotation(self.cfg.dataset_dir, img_name, self._class_to_ind)
        # annotation = np.concatenate( (annotation['boxes'], annotation['gt_classes'].reshape(-1, 1) ), axis=1)
        # for i in range(len(annotation)):
        #     self.truth[img_name].append(annotation[i])

        bboxes = np.array(self.truth.get(img_name), dtype=np.float)
        img_path = os.path.join(self.cfg.dataset_dir, 'VOC2007/JPEGImages', img_name+'.jpg')
        img = cv2.imread(img_path)
        # 保持图像原尺寸
        self.cfg.w = img.shape[1]
        self.cfg.h = img.shape[0]

        use_mixup = self.cfg.mixup
        # if random.randint(0, 1):    # 50%的几率执行Mosic
        #     use_mixup = 0
        #     print('Image ({}) don\'t run Mosaic'.format(img_name))

        if use_mixup == 3:
            min_offset = 0.2
            cut_x = random.randint(int(self.cfg.w * min_offset), int(self.cfg.w * (1 - min_offset)))
            cut_y = random.randint(int(self.cfg.h * min_offset), int(self.cfg.h * (1 - min_offset)))

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        gaussian_noise = 0

        out_img = np.zeros([self.cfg.h, self.cfg.w, 3])
        out_bboxes = []

        for i in range(use_mixup + 1):
            if i != 0:
                img_name = random.choice(list(self.truth.keys()))
                bboxes = np.array(self.truth.get(img_name), dtype=np.float)
                img_path = os.path.join(self.cfg.dataset_dir, 'VOC2007/JPEGImages', img_name+'.jpg')
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            oh, ow, oc = img.shape
            dh, dw, dc = np.array(np.array([oh, ow, oc]) * self.cfg.jitter, dtype=np.int)

            dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
            dsat = rand_scale(self.cfg.saturation)
            dexp = rand_scale(self.cfg.exposure)

            pleft = random.randint(-dw, dw)
            pright = random.randint(-dw, dw)
            ptop = random.randint(-dh, dh)
            pbot = random.randint(-dh, dh)

            flip = random.randint(0, 1) if self.cfg.flip else 0

            if (self.cfg.blur):
                tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                if tmp_blur == 0:
                    blur = 0
                elif tmp_blur == 1:
                    blur = 1
                else:
                    blur = self.cfg.blur

            if self.cfg.gaussian and random.randint(0, 1):
                gaussian_noise = self.cfg.gaussian
            else:
                gaussian_noise = 0

            if self.cfg.letter_box:
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            truth, min_w_h = fill_truth_detection(bboxes, self.cfg.boxes, self.cfg.classes, flip, pleft, ptop, swidth,
                                                  sheight, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            ai = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                         dhue, dsat, dexp, gaussian_noise, blur, truth)

            if use_mixup == 0:
                out_img = ai
                out_bboxes = truth
            if use_mixup == 1:
                if i == 0:
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    out_img = cv2.addWeighted(ai, 0.5, old_img, 0.5)
                    out_bboxes = np.concatenate([old_truth, truth], axis=0)
            elif use_mixup == 3:
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = int(min(cut_x, max(0, (-int(pleft) * self.cfg.w / swidth))))
                top_shift = int(min(cut_y, max(0, (-int(ptop) * self.cfg.h / sheight))))

                right_shift = int(min((self.cfg.w - cut_x), max(0, (-int(pright) * self.cfg.w / swidth))))
                bot_shift = int(min(self.cfg.h - cut_y, max(0, (-int(pbot) * self.cfg.h / sheight))))

                out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth.copy(), self.cfg.w, self.cfg.h, cut_x,
                                                       cut_y, i, left_shift, right_shift, top_shift, bot_shift)
                out_bboxes.append(out_bbox)
                # print(img_path)
        if use_mixup == 3:
            out_bboxes = np.concatenate(out_bboxes, axis=0)
        out_bboxes1 = np.zeros([self.cfg.boxes, 5])
        out_bboxes1[:min(out_bboxes.shape[0], self.cfg.boxes)] = out_bboxes[:min(out_bboxes.shape[0], self.cfg.boxes)]
        return out_img, out_bboxes1

    def _get_val_item(self, index):
        """
        """
        img_path = self.imgs[index]
        bboxes_with_cls_id = np.array(self.truth.get(img_path), dtype=np.float)
        img = cv2.imread(os.path.join(self.cfg.dataset_dir, img_path))
        # img_height, img_width = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.cfg.w, self.cfg.h))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        num_objs = len(bboxes_with_cls_id)
        target = {}
        # boxes to coco format
        boxes = bboxes_with_cls_id[...,:4]
        boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]  # box width, box height
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(bboxes_with_cls_id[...,-1].flatten(), dtype=torch.int64)
        target['image_id'] = torch.tensor([get_image_id(img_path)])
        target['area'] = (target['boxes'][:,3])*(target['boxes'][:,2])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        return img, target


def get_image_id(filename:str) -> int:
    """
    Convert a string to a integer.
    Make sure that the images and the `image_id`s are in one-one correspondence.
    There are already `image_id`s in annotations of the COCO dataset,
    in which case this function is unnecessary.
    For creating one's own `get_image_id` function, one can refer to
    https://github.com/google/automl/blob/master/efficientdet/dataset/create_pascal_tfrecord.py#L86
    or refer to the following code (where the filenames are like 'level1_123.jpg')
    >>> lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    >>> lv = lv.replace("level", "")
    >>> no = f"{int(no):04d}"
    >>> return int(lv+no)
    """
    # raise NotImplementedError("Create your own 'get_image_id' function")
    # lv, no = os.path.splitext(os.path.basename(filename))[0].split("_")
    # lv = lv.replace("level", "")
    # no = f"{int(no):04d}"
    # return int(lv+no)

    print("You could also create your own 'get_image_id' function.")
    # print(filename)
    parts = filename.split('/')
    id = int(parts[-1][0:-4])
    # print(id)
    return id

#

if __name__ == "__main__":
    from Mosaic_cfg import Cfg
    import matplotlib.pyplot as plt
    from xml_create import xml_save
    from tqdm import tqdm

    random.seed(2020)
    np.random.seed(2020)



    '''1 debug'''
    # Cfg.dataset_dir = '/media/4TDisk/zzl/datasets/Github_data/cityscape_fake_cyclegan_with_foggy'
    # imgs_txt = './temp.txt'
    # print('----------')
    # print(imgs_txt)
    # dataset = Yolo_dataset(imgs_txt, Cfg)
    #
    # img_out_path = './out6'
    # annotation_out_path = './out6'
    # if not os.path.exists(img_out_path):
    #     os.makedirs(img_out_path)
    # if not os.path.exists(annotation_out_path):
    #     os.makedirs(annotation_out_path)
    # print('img_out_path:{}'.format(img_out_path))
    # print('annotation_out_path:{}'.format(annotation_out_path))
    # for i in tqdm(range(len(dataset))):
    #     out_img, out_bboxes = dataset.__getitem__(i)
    #     # remove the invalid boxes
    #     out_bboxes = np.delete(out_bboxes, [t_i for t_i, temp in enumerate(out_bboxes) if temp[-1] == 0], axis=0)
    #     a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))
    #
    #     # plt.imshow(a.astype(np.int32))
    #     # plt.show()
    #     # print(os.path.abspath('./out'))
    #
    #
    #     # print(os.path.abspath(out_path))
    #
    #     cv2.imwrite(img_out_path + '/' + dataset.imgs[i] + '_mosaic.jpg', out_img)
    #     cv2.imwrite(img_out_path + '/' + dataset.imgs[i] + '_mosaic_labeled.jpg', a)
    #
    #     objects_bndbox, objects_class = np.split(out_bboxes, [4], 1)
    #     objects_class = np.array([dataset._classes[int(temp)] for temp1 in objects_class for temp in temp1]).reshape(-1,
    #                                                                                                                  1)
    #     objects_bndbox = np.array([str(int(temp)) for temp1 in objects_bndbox for temp in temp1]).reshape(-1, 4)
    #     objects_save = np.concatenate((objects_bndbox, objects_class), axis=1)
    #
    #     xml_save(save_path=annotation_out_path,
    #              filename=dataset.imgs[i] + '_mosaic',
    #              size=[str(temp) for temp in out_img.shape],
    #              objects=objects_save.tolist())

    '''2 cityscape_fake_cyclegan_with_foggy'''
    Cfg.dataset_dir = '/media/4TDisk/zzl/datasets/Github_data/cityscape_fake_cyclegan_with_foggy'
    imgs_txt = '/media/4TDisk/zzl/datasets/Github_data/cityscape_fake_cyclegan_with_foggy/VOC2007/ImageSets/Main/train_combine_fg.txt'
    print('----------')
    print(imgs_txt)
    dataset = Yolo_dataset(imgs_txt, Cfg)

    img_out_path = '/media/4TDisk/zzl/datasets/Github_data/cityscape_fake_cyclegan_with_foggy/VOC2007/JPEGImages'
    annotation_out_path = '/media/4TDisk/zzl/datasets/Github_data/cityscape_fake_cyclegan_with_foggy/VOC2007/Annotations'

    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)
    if not os.path.exists(annotation_out_path):
        os.makedirs(annotation_out_path)
    print('img_out_path:{}'.format(img_out_path))
    print('annotation_out_path:{}'.format(annotation_out_path))
    for i in tqdm(range(len(dataset))):
        out_img, out_bboxes = dataset.__getitem__(i)
        # remove the invalid boxes
        out_bboxes = np.delete(out_bboxes, [t_i for t_i, temp in enumerate(out_bboxes) if temp[-1] == 0], axis=0)
        # a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))

        # plt.imshow(a.astype(np.int32))
        # plt.show()
        # print(os.path.abspath('./out'))


        # print(os.path.abspath(out_path))

        cv2.imwrite(img_out_path + '/' + dataset.imgs[i] + '_mosaic.jpg', out_img)
        # cv2.imwrite(out_path + '/' + dataset.imgs[i] + '_mosaic_labeled.jpg', a)

        objects_bndbox, objects_class = np.split(out_bboxes, [4], 1)
        objects_class = np.array([dataset._classes[int(temp)] for temp1 in objects_class for temp in temp1]).reshape(-1,
                                                                                                                     1)
        objects_bndbox = np.array([str(int(temp)) for temp1 in objects_bndbox for temp in temp1]).reshape(-1, 4)
        objects_save = np.concatenate((objects_bndbox, objects_class), axis=1)

        xml_save(save_path=annotation_out_path,
                 filename = dataset.imgs[i] + '_mosaic',
                 size = [str(temp) for temp in out_img.shape],
                 objects = objects_save.tolist())



    '''3 foggy_fake_cyclegan_with_cityscape'''
    Cfg.dataset_dir = '/media/4TDisk/zzl/datasets/Github_data/foggy_fake_cyclegan_with_cityscape'
    imgs_txt = '/media/4TDisk/zzl/datasets/Github_data/foggy_fake_cyclegan_with_cityscape/VOC2007/ImageSets/Main/train_combine_cs.txt'
    print('----------')
    print(imgs_txt)
    dataset = Yolo_dataset(imgs_txt, Cfg)

    img_out_path = '/media/4TDisk/zzl/datasets/Github_data/foggy_fake_cyclegan_with_cityscape/VOC2007/JPEGImages'
    annotation_out_path = '/media/4TDisk/zzl/datasets/Github_data/foggy_fake_cyclegan_with_cityscape/VOC2007/Annotations'
    # out_path = os.path.join('./out5')
    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)
    if not os.path.exists(annotation_out_path):
        os.makedirs(annotation_out_path)
    print('img_out_path:{}'.format(img_out_path))
    print('annotation_out_path:{}'.format(annotation_out_path))
    for i in tqdm(range(len(dataset))):
        out_img, out_bboxes = dataset.__getitem__(i)
        # remove the invalid boxes
        out_bboxes = np.delete(out_bboxes, [t_i for t_i, temp in enumerate(out_bboxes) if temp[-1] == 0], axis=0)
        # a = draw_box(out_img.copy(), out_bboxes.astype(np.int32))

        # plt.imshow(a.astype(np.int32))
        # plt.show()
        # print(os.path.abspath('./out'))


        # print(os.path.abspath(out_path))

        cv2.imwrite(img_out_path + '/' + dataset.imgs[i] + '_mosaic.jpg', out_img)
        # cv2.imwrite(out_path + '/' + dataset.imgs[i] + '_mosaic_labeled.jpg', a)

        objects_bndbox, objects_class = np.split(out_bboxes, [4], 1)
        objects_class = np.array([dataset._classes[int(temp)] for temp1 in objects_class for temp in temp1]).reshape(-1,
                                                                                                                     1)
        objects_bndbox = np.array([str(int(temp)) for temp1 in objects_bndbox for temp in temp1]).reshape(-1, 4)
        objects_save = np.concatenate((objects_bndbox, objects_class), axis=1)

        xml_save(save_path=annotation_out_path,
                 filename=dataset.imgs[i] + '_mosaic',
                 size=[str(temp) for temp in out_img.shape],
                 objects=objects_save.tolist())