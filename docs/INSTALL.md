# Installation

## Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04/22.04)
* Python 3.9+
* [PyTorch <= 1.10](https://pytorch.org/get-started/previous-versions/) [(Deprecation of THC.h in PyTorch >=1.11.*)](https://github.com/open-mmlab/mmdetection3d/issues/1332)
* CUDA >= 10.0
* [`spconv v2.x`](https://github.com/traveller59/spconv)

## Install `coda-models`

a. Clone this repository.
```bash
git clone git@github.com:ut-amrl/coda-models.git
```

b. Install the dependent libraries as follows

```
conda env create -f environment.yml
```

If the above command fails, you can also run the following commands in order:

<b>ROS support (optional)</b>
```
conda install mamba -c conda-forge
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
conda install ros-noetic-desktop
```

<b>Model Inference (mandatory)</b>
```
conda install -c "nvidia/label/cuda-11.3.1" cuda

conda install -c conda-forge cudatoolkit-dev==11.3.1

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install spconv-cu113

pip install -r requirements.txt
```

c. Link to cuda_11.3 on your machine and use correct GCC/G++

```bashrc
source cuda_11.3_env 

sudo apt-get install gcc-9 g++-9 -y
sudo ln -sfn /usr/bin/gcc-9 /usr/bin/gcc
sudo ln -sfn /usr/bin/gcc-9 /usr/bin/cc
sudo ln -sfn /usr/bin/g++-9 /usr/bin/g++
sudo ln -sfn /usr/bin/g++-9 /usr/bin/c++
```

d. Install this repository

```
python setup.py develop
```

## Post Installation Steps

Verify your installation by following any of the tutorials in the [Getting Started](./GETTING_STARTED.md)
section.
