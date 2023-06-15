# Test model
# CUDA_VISIBLE_DEVICES=0 python faster_rcnn_test.py --dataset cityscape --model_dir './output/model/vgg16/cityscape/faster_rcnn_1_1_5931.pth' --cuda


for epoch in 6 7 8 9 10
do
    echo "The epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python faster_rcnn_test.py --dataset cityscape --model_dir ./output/model/vgg16/cityscape/faster_rcnn_1_${epoch}_5931.pth --cuda
done

echo `date`
