M="ATF"
printf "Training mission: %s start!\n" ${M}

CUDA_VISIBLE_DEVICES=0 python ATF_train.py \
                              --dataset cityscape \
                              --net vgg16 \
                              --save_dir ./output/da_model \
                              --epochs 14 \
                              --bs 1 \
                              --lr 1e-3 \
                              --lr_decay_step 10 \
                              --cuda \
                              --Mission ${M} \
                              --disp_interval 100

echo `date`
printf "\n Mission: %s is over!\n" ${M}