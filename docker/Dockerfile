ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade numpy

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
#RUN pip install --no-cache-dir --upgrade pip wheel setuptools
#RUN pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

# install MMENGINE
#RUN pip3 install --ignore-installed PyYAML

#RUN pip3 install pip==8.1.1
#RUN pip3 uninstall PyYAML
#RUN pip install PyYAML==8.1.1

#RUN conda remove PyYAML
#RUN pip install -U openmim
#RUN mim install mmengine
#RUN mim install "mmcv>=2.0.1"
#RUN mim install "mmdet>=3.1.0"


# Install MMPose

#RUN conda clean --all


RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"



WORKDIR /mmpose
RUN mkdir -p /mmpose/data
ENV FORCE_CUDA="1"

# original
#RUN pip install -r requirements/build.txt
#RUN pip install --no-cache-dir -e .

RUN pip install -r /mmpose/requirements.txt
RUN pip install -v -e .