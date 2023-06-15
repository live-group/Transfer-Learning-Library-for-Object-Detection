#!/bin/bash


M="ATF_test"
printf "Test mission: %s start!\n" ${M}
#

for epoch in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    echo "The epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python ATF_test.py --dataset cityscape --net vgg16 --part test_t --model_dir ./output/da_model/2nd/vgg16/cityscape/cityscape_1_${epoch}.pth --cuda
done

echo `date`
printf "\n Mission: %s is over!\n" ${M}

