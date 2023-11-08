#FROM airlab404/dl:cuda10_pytorch_py36
FROM airlab404/dl:cuda11.1_pytorch-1.8.1_conda_py38

# Install extras
#COPY docker/requirements.yml /requirements.yml

# useful only for projects need torchlight
#COPY ./torchlight /torchlight
#WORKDIR /
#RUN pip install -e torchlight

RUN pip install torch==2.0.1 torchvision==0.15.2

RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"
RUN mim install "mmdet>=3.1.0"

RUN pip install albumentations
RUN pip install opencv-python-headless==4.1.2.30

#WORKDIR ./mmpose
#COPY  docker/requirements.txt /requirements.txt
COPY ./mmpose /mmpose
#RUN chmod -R 777 /mmpose
WORKDIR /mmpose
RUN ls
RUN pip install -r requirements.txt
RUN pip install -v -e .

WORKDIR /mmpose

# If you are using a py27 image, change this to py27
#RUN /bin/bash -c ". activate py38 && conda env update -f=/requirements.yml"

CMD ["bash"]

WORKDIR /mmpose