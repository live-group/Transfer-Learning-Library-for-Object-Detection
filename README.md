# Transfer-Learning-Library-for-Object-Detection
Transfer Learning Library for Domain Adaptation and Domain Generalization of Object Detection.


## Introduction
This is a Transfer Learning library for Object Detection task. It is mainly implemented by PyTorch.  

We provide the implementation of many methods in the directory _methods_. They belong to several different learning setups, including:  
 * UDAOD (Unsupervised Domain Adaptive Object Detection)  
 * DGOD (Domain Generalized Object Detection)
 * UniDAOD (Universal Domain Adaptive Object Detection)  

## Provided Methods
The currently provided methods include:  

##### Unsupervised Domain Adaptive Object Detection
* **DAF** -Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR 2018]](https://arxiv.org/abs/1803.03243)[[local code]](/methods/DAF)[[corresponding original code]](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)
* **MAF** -Multi-adversarial Faster-RCNN for Unrestricted Object Detection [[ICCV 2019]](https://arxiv.org/abs/1907.10343)[[local code]](/methods/MAF)[[corresponding original code]](https://github.com/He-Zhenwei/MAF)
* **ATF** -Domain Adaptive Object Detection via Asymmetric Tri-way Faster-RCNN [[ECCV 2020]](http://arxiv.org/abs/2007.01571)[[local code]](/methods/ATF)[[corresponding original code]](https://github.com/He-Zhenwei/ATF)
* **IDF** -Exploring Implicit Domain-invariant Features for Domain Adaptive Object Detection [[TCSVT 2023]](https://ieeexplore.ieee.org/document/9927485)[[local code]](/methods/IDF)[[corresponding original code]](https://github.com/sea123321/IDF)
* **PA-ATF** -Partial Alignment for Object Detection in the Wild [[TCSVT 2022]](https://ieeexplore.ieee.org/document/9663266)[[local code]](/methods/PA_ATF)
* **PT-MAF** -Multi-adversarial Faster-RCNN with Paradigm Teacher for Unrestricted Object Detection [[IJCV 2022]](https://link.springer.com/article/10.1007/s11263-022-01728-z)[[local code]](/methods/PT_MAF)

##### Domain Generalized Object Detection
* **MAD** -Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains [[CVPR 2023]](https://arxiv.org/abs/2304.02950)[[local code]](/methods/MAD)[[corresponding original code]](https://github.com/K2OKOH/MAD)

##### Universal Domain Adaptive Object Detection
* **US-DAF** -Universal Domain Adaptive Object Detector [[ACM MM 2022]](http://arxiv.org/abs/2207.01756)[[local code]](/methods/US_DAF)[[corresponding original code]](https://github.com/a-shi321/US-DAF)

## Usage

#### Data Preparation
All datasets are aranged in the format of PASCAL VOC as follows:
```shell
# cityscapes   
- cityscape
    - VOC2007
        - ImageSets  
        - JPEGImages  
        - Annotations  
```

#### Pretrained Model
We use two pretrained models, VGG16 and ResNet101. You can download these two models from:  
* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth), [Baidu Netdisk](https://pan.baidu.com/s/1uhLiHVbPL78goJVZg67Ifw?pwd=1oo0)  
* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth), [Baidu Netdisk](https://pan.baidu.com/s/1tlvAaKgyuKR-iO0JKmYh0Q?pwd=g1ln)  

#### Environment Configuration
The pytorch environment includes python 3.6, pytorch 0.4.0, CUDA 8.0 or higher, torchvision 0.2.1, matplotlib 3.3.4, etc. See the *requirements.txt* file for details.  
The following is a simple example of configuring an environment using conda.  
```shell
# create environment
conda create --name torch0_4 python=3.6
# activate environment
conda activate torch0_4
# install pytorch
conda install pytorch=0.4.0 cuda80 -c pytorch
# install other requirements
pip install -r requirements.txt
```
The default version we provide is compiled with Python 3.6, so if you're using another version of Python, use the following command to compile it yourself:  
```shell
# compile the cuda dependencies
cd lib
sh make.sh
```

#### Train and Test
You can find methods in the directory *methods*. In general, there are four main files in each method directory, which are:  
*_train.py* - Training script for the model  
*_test.py* - Test script for the model  
*_train.sh* - Detailed model train run commands   
*_test.sh* - Detailed model test run commands  

#### Results
   
##### Unsupervised Domain Adaptive Object Detection
* **Cityscapes** (source) -> **Foggy cityscapes** (target)  backbone: **VGG16**  

|            |   person |   rider |   car  |   truck |    bus |   train |   motorcycle |   bicycle |    mAP |
|:----------:|:--------:|:-------:|:------:|:-------:|:------:|:-------:|:------------:|:---------:|:------:|
| DAF        | 29.9     | 41.2    | 43.3   | 20.2    | 36.3   | 27.6    | 26.4         | 33.4      | 32.3   |
| MAF        | 33.2     | 44.9    | 44.3   | 28.5    | 40.1   | 23.9    | 30.9         | 37.4      | 35.4   |
| ATF        | 33.8     | 46.7    | 44.9   | 26.5    | 45.5   | 32.5    | 34.9         | 38.2      | 37.9   |
| IDF        | 36.4     | 48.6    | 52.4   | 33.9    | 52.3   | 35.2    | 36.9         | 39.6      | 41.9   |
| PA-ATF     | 34.3     | 45.6    | 52.1   | 28.7    | 47.5   | 49.4    | 33.7         | 37.4      | 41.1   |
| PT-MAF     | 34.2     | 50.4    | 50.0   | 27.3    | 47.2   | 46.1    | 32.5         | 38.5      | 40.8   |

##### Domain Generalized Object Detection
* Training Dataset:**Cityscapes**, Test Dataset:**Foggy cityscapes**  backbone: **VGG16**    

|            |   person |   rider |   car  |   truck |    bus |   train |   motorcycle |   bicycle |    mAP |
|:----------:|:--------:|:-------:|:------:|:-------:|:------:|:-------:|:------------:|:---------:|:------:|
| MAD        | 33.9     | 46.9    | 45.1   | 28.0    | 44.1   | 34.4    | 33.7         | 39.3      | 38.2   |

##### Universal Domain Adaptive Object Detection
* **VOC** -> **Clipart** (The two datasets share 10 categories)  backbone: **ResNet101**   

|            |   bus    |   car   |   cat  |   chair |    cow |   diningtable |   dog  |   horse |   motorbike |   person |    mAP |
|:----------:|:--------:|:-------:|:------:|:-------:|:------:|:-------------:|:------:|:-------:|:-----------:|:--------:|:------:|
| US-DAF     | 31.1     | 43.2    | 13.9   | 39.4    | 58.7   | 27.4          | 12.6   | 44.3    | 55.4        | 61.3     | 38.7   |

## Contact
If you have any problem about our code, feel free to contact us.  
* Zhilong Zhang (zhangzhilong@stu.cqu.edu.cn)
* Lei Zhang (leizhang@cqu.edu.cn)

or describe it in Issues.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.  
```latex
@misc{lib,
    author = {Zhilong Zhang, Lei Zhang},
    title = {Transfer-Learning-Library-for-Object-Detection},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/live-group/Transfer-Learning-Library-for-Object-Detection}},
}
```

## Acknowledgment
The base object detector in this library is built upon [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), and we'd like to appreciate for their excellent works.  




