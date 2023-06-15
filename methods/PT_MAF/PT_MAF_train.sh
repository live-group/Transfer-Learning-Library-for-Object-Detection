
# 1.use the labeled source domain data to train a faster rcnn model
# then, ori_model = 'model_path' in PT-MAF_train.py line 385

cd ../faster_rcnn
CUDA_VISIBLE_DEVICES=0 python faster_rcnn_train.py --dataset cityscape --net vgg16 --bs 1 --lr 2e-3 --lr_decay_step 6 --epochs 10 --cuda


# 2. train PT-MAF model
CUDA_VISIBLE_DEVICES=0 python import PT-MAF_train.py --dataset cityscape --net vgg16 --save_dir ./output/da_model --epochs 10 --bs 1 --lr 2e-3 --lr_decay_step 6 --cuda --disp_interval 100


:<<!
M="PT-MAF"
printf "Training mission: %s start!\n" ${M}

CUDA_VISIBLE_DEVICES=0 python PT_MAF_train.py \
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
!




