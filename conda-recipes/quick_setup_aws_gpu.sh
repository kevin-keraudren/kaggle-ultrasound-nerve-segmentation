#!/bin/bash

set -x
set -e

# Anaconda set-up for https://aws.amazon.com/marketplace/pp/B00FYCDDTE
# Amazon Linux AMI with NVIDIA GRID GPU Driver
# g2.x2large has compute capability 3.0
# US East (N. Virginia)	ami-2e5e9c43

# https://github.com/jjhelmus/wip_conda_recipes/tree/master/tensorflow
# http://www.bazel.io/docs/install.html#ubuntu

# Launch from EC2 console (not from the market place) and 
# ensure that we have the `/media/ephemeral0/` storage (Instance Store 0):
# you are entitled to 60GB, but you do not get them by default!

# do not let `/tmp` eat up the small boot disk, 
# send it to `/media/ephemeral0/` instead.
# https://gist.github.com/erikbern/78ba519b97b440e10640
sudo mkdir /media/ephemeral0/tmp
sudo chmod 777 /media/ephemeral0/tmp
sudo rm -rf /tmp
sudo ln -s /media/ephemeral0/tmp /tmp

# install anaconda on `ephemeral0`
sudo mkdir /media/ephemeral0/anaconda
sudo chmod 777 /media/ephemeral0/anaconda

# Install Anaconda
ANACONDA_VERSION=4.1.1
echo "Installing Anaconda ${ANACONDA_VERSION}"
curl -s -L -O http://repo.continuum.io/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh && \
    bash Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -f -p /media/ephemeral0/anaconda
echo export PATH="/media/ephemeral0/anaconda/bin:\$PATH" >> $HOME/.bashrc
source $HOME/.bashrc
rm Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh

# Install Python packages
conda install -y h5py graphviz libtool git

conda install -y -c memex opencv
pip install scipy --upgrade
pip install pydot-ng seaborn

## cudnnn
conda install -y -c ostrokach cudnn=4.0

## tensorflow
sudo rm -rf /usr/local/include
sudo ln -s /media/ephemeral0/anaconda/include /usr/local/include
sudo rm -rf /usr/local/lib
sudo ln -s /media/ephemeral0/anaconda/lib /usr/local/lib

sudo ln -s /opt/nvidia/cuda/ /usr/local/cuda
sudo cp /usr/local/include/cudnn.h /usr/local/cuda/include
sudo cp /usr/local/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

conda install -y -c kevin-keraudren tensorflow

## Install latest Keras
pip install git+https://github.com/fchollet/keras.git --upgrade
echo export KERAS_BACKEND=tensorflow >> ~/.bashrc

# Do some cleaning
sudo yum clean all
conda clean -all -y

## Install the necessary tools to survive in a terminal
sudo yum -y install emacs lynx htop

## using iterm and imgcat, we can look at images straight from the terminal
cd /media/ephemeral0/anaconda/bin/
wget https://raw.githubusercontent.com/gnachman/iTerm2/master/tests/imgcat
chmod +x imgcat

# get lynx config file
cd
wget https://raw.githubusercontent.com/kevin-keraudren/conda-recipes/master/.lynxrc

cd /tmp

echo "Now get the data!"
echo lynx -cfg=~/.lynxrc https://www.kaggle.com/c/ultrasound-nerve-segmentation/data
echo mkdir -p /tmp/data && unzip train.zip -d data
echo mkdir -p /tmp/data && unzip test.zip -d data
