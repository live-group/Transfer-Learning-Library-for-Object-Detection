#!/bin/bash
# training script
# train on two data set, one is the original data set(s1), another is an augmented version(s2).
# ['voc', 'cityscape', 'kitti', 'watercolor', 'clipart', 'sim10k', 'kitti', 'bdd100k', 'rain', 'foggy']
M="MV3"
printf "Training mission: %s start!\n" ${M}

CUDA_VISIBLE_DEVICES=0 python MAD_train.py \
        --dataset       dg_union \
        --net           vgg16 \
        --cuda          \
        --epochs        10 \
        --bs            1 \
        --save_dir      ./SaveFile/model \
        --Mission       ${M} \
        --mode          train_model \
        --log_flag      1 \
        --lr            2e-3 \
        --lr_decay_step 6 \
        \
        --T_Set         foggy \
        --T_Part        test \
        --T_Type        s1 \
        \
        --S1_Set        cityscape \
        --S1_Part       train \
        --S1_Type       s1 \
        \
        --S2_Set        cityscape \
        --S2_Part       train \
        --S2_Type       s2 \
        \

echo `date`
printf "\n Mission: %s is over!\n" ${M}
