#!/usr/bin/env bash

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export CPATH=/usr/local/cuda-8.0/include${CPATH:+:${CPATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda-9.1/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

python setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_61,code=sm_61 "

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH

cd ../
python build.py

# compile roi_pooling
cd ../../
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_align
cd ../../
cd model/roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_crop
cd ../../
cd model/roi_crop/src
echo "Compiling roi crop kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

