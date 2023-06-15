###### Test faster_rcnn model
# CUDA_VISIBLE_DEVICES=0 python faster_rcnn_test.py --dataset foggy_cityscape --dataset_part test --model_dir './output/model/vgg16/cityscape/faster_rcnn_1_1_5931.pth' --cuda

:<<!
train_dataset='cs_cyclegan_fg'
for epoch in 1 2 3 4 5 6 7 8 9 10
do
    echo "The epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python faster_rcnn_test.py --dataset foggy_cityscape --dataset_part test --model_dir ./output/model/vgg16/${train_dataset}/faster_rcnn_1_${epoch}_5931.pth --cuda
done

echo `date`
!

##### Test IDF model
# CUDA_VISIBLE_DEVICES=0 python IDF_test.py --dataset_t cs_fg --net vgg16  --load_name ./output/model/vgg16/cs_combine_fg2cs_fg_combine_cs/faster_rcnn_1_7_5931.pth


#:<<!
train_dataset='cs_combine_fg_combine_mosaic2cs_fg_combine_cs_combine_mosaic'
for epoch in 1 2 3 4 5 6 7 8 9 10
do
    echo "The epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python IDF_test.py --dataset_t cs_fg --net vgg16  --load_name ./output/model/vgg16/${train_dataset}/DA_ObjectDetection_session_1_epoch_${epoch}_step_10000.pth
done
#!

echo `date`
