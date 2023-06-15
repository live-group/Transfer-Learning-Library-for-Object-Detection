#!/bin/bash
#

M="MV3_test"
printf "Test mission: %s start!\n" ${M}
# # ['voc', 'cityscape', 'kitti', 'watercolor', 'clipart', 'sim10k', 'kitti', 'bdd100k', 'rain', 'foggy']

for epoch in 6 7 8 9 10
do
    echo "The epoch is: $epoch"

    CUDA_VISIBLE_DEVICES=0 python MAD_test.py \
    --net           vgg16 \
    --cuda          \
    --model_dir     "SaveFile/model/vgg16-cityscape-cityscape/MV3/model_e$epoch.pth" \
    --dataset       dg_union \
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


done

echo `date`
printf "\n Mission: %s is over!\n" ${M}


