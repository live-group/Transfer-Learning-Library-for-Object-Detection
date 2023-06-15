

# CUDA_VISIBLE_DEVICES=0 python US_DAF_test.py --dataset VOC2clipart --part test_t --model_dir ./output/da_model/open_set0_5/res101/VOC2clipart/voc2clipart_1_1_10000.pth --cuda


for epoch in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    echo "the epoch is: ${epoch}"
    CUDA_VISIBLE_DEVICES=0 python US_DAF_test.py \
    --dataset VOC2clipart --part test_t \
    --model_dir ./output/da_model/open_set0_5/res101/VOC2clipart/voc2clipart_1_${epoch}_10000.pth \
    --cuda
done





