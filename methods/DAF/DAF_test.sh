# Test model in target domain
CUDA_VISIBLE_DEVICES=0 python DAF_test.py --dataset cityscape --part test_t --model_dir='./output/model_weight/vgg16/cityscape/model_e6.pth' --cuda
CUDA_VISIBLE_DEVICES=0 python DAF_test.py --dataset cityscape --part test_t --model_dir='./output/model_weight/vgg16/cityscape/model_e7.pth' --cuda
CUDA_VISIBLE_DEVICES=0 python DAF_test.py --dataset cityscape --part test_t --model_dir='./output/model_weight/vgg16/cityscape/model_e8.pth' --cuda
CUDA_VISIBLE_DEVICES=0 python DAF_test.py --dataset cityscape --part test_t --model_dir='./output/model_weight/vgg16/cityscape/model_e9.pth' --cuda
CUDA_VISIBLE_DEVICES=0 python DAF_test.py --dataset cityscape --part test_t --model_dir='./output/model_weight/vgg16/cityscape/model_e10.pth' --cuda
