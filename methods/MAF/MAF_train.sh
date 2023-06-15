# CUDA_VISIBLE_DEVICES=2 python MAF_train.py --dataset cityscape --net vgg16 --save_dir ./output/da_model --epochs 10 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda --disp_interval 100

M="MAF"
printf "Training mission: %s start!\n" ${M}

CUDA_VISIBLE_DEVICES=0 python MAF_train.py \
                              --dataset cityscape \
                              --net vgg16 \
                              --save_dir ./output/da_model \
                              --epochs 10 \
                              --bs 1 \
                              --lr 2e-3 \
                              --lr_decay_step 6 \
                              --cuda \
                              --Mission ${M} \
                              --disp_interval 100

echo `date`
printf "\n Mission: %s is over!\n" ${M}






