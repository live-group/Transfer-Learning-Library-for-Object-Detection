
# CUDA_VISIBLE_DEVICES=0 python MAF_test.py --dataset cityscape --net vgg16 --part test_t --model_dir ./output/da_model/vgg16/cityscape/cityscape_1_13.pth --cuda


for epoch in 6 7 8 9 10
do
    echo "The epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python MAF_test.py --dataset cityscape --net vgg16 --part test_t --model_dir ./output/da_model/vgg16/cityscape/cityscape_1_${epoch}.pth --cuda
done


