# Select the base image
FROM nvcr.io/nvidia/pytorch:21.10-py3

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
RUN pip install opencv-python==4.5.5.64
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install networkx==3.0 lark-parser joblib scikit-learn torchsummary setuptools tensorboard numpy>=1.18.5 tqdm>=4.41.0 matplotlib>=3.2.2 opencv-python>=4.1.2 Pillow PyYAML>=5.3.1 scipy>=1.4.1 seaborn pandas rtpt

pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install torch-geometric
pip install wandb anytree