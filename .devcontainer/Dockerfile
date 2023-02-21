# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.07-py3
FROM nvcr.io/nvidia/pytorch:21.10-py3

#FROM anibali/pytorch:1.8.0-cuda10.2-ubuntu20.04

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

# Select the working directory
WORKDIR  /Workspace

# Install system libraries required by OpenCV.
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*



RUN pip install --upgrade pip
# Install Python requirements
# COPY requirements.txt /tmp/requirements.txt
#RUN python -m pip install -r /tmp/requirements.txt
# RUN pip install -r requirements.txt

#RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install opencv-python==4.5.5.64
# COPY ./requirements.txt ./requirements.txt
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install lark-parser sklearn torchsummary setuptools tensorboard numpy>=1.18.5 tqdm>=4.41.0 matplotlib>=3.2.2 opencv-python>=4.1.2 Pillow PyYAML>=5.3.1 scipy>=1.4.1 seaborn pandas rtpt
# RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html


# install PyG
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
# pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
RUN pip install torch-geometric

RUN pip install opencv-python==4.5.5.64
