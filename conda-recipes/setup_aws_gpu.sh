#!/bin/bash

# Anaconda set-up for https://aws.amazon.com/marketplace/pp/B00FYCDDTE
# Amazon Linux AMI with NVIDIA GRID GPU Driver
# g2.x2large has compute capability 3.0

# https://github.com/jjhelmus/wip_conda_recipes/tree/master/tensorflow
# http://www.bazel.io/docs/install.html#ubuntu

# Launch from EC2 console (not from the market place) and 
# ensure that we have the `/media/ephemeral0/` storage (Instance Store 0):
# you are entitled to 60GB, but you do not get them by default!

# check device mapping
curl http://169.254.169.254/latest/meta-data/block-device-mapping/

# view volume mapped but not formatted nor mounted
lsblk

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

# Install JDK8 (required for bazel, required to build tensorflow)
cd /opt/
sudo wget --no-cookies --no-check-certificate --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com%2F; oraclelicense=accept-securebackup-cookie" "http://download.oracle.com/otn-pub/java/jdk/8u101-b13/jdk-8u101-linux-x64.tar.gz"
sudo tar xzf jdk-8u101-linux-x64.tar.gz
sudo rm jdk-8u101-linux-x64.tar.gz
echo export JAVA_HOME=/opt/jdk1.8.0_101 >> $HOME/.bashrc
echo export JRE_HOME=/opt/jdk1.8.0_101/jre >> $HOME/.bashrc
echo export PATH=$PATH:/opt/jdk1.8.0_101/bin:/opt/jdk1.8.0_101/jre/bin >> $HOME/.bashrc
source $HOME/.bashrc

# install bazel
cd /tmp
wget https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-linux-x86_64.sh
bash bazel-0.3.1-installer-linux-x86_64.sh --user
rm bazel-0.3.1-installer-linux-x86_64.sh

# Install Anaconda
ANACONDA_VERSION=4.1.1
echo "Installing Anaconda ${ANACONDA_VERSION}"
curl -s -L -O http://repo.continuum.io/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh && \
    bash Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -f -p /media/ephemeral0/anaconda
echo export PATH="/media/ephemeral0/anaconda/bin:\$PATH" >> $HOME/.bashrc
source $HOME/.bashrc
rm Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh

# Install Python packages
conda install h5py graphviz libtool git

conda install -c memex opencv
pip install scipy --upgrade
pip install pydot-ng seaborn

# Note:
# conda install -c menpo opencv3
# leads to https://askubuntu.com/questions/761589/installing-libgtk-x11-2-0-so-0-in-ubuntu-15-04
# Welcome to Amazon Linux! Hence the choice of `-c memex` which happens to work fine.

## cudnnn
conda install -c ostrokach cudnn=4.0

# protobuf
conda build protobuf --no-activate
anaconda upload /media/ephemeral0/anaconda/conda-bld/linux-64/protobuf-3.0.0b2-py35_1.tar.bz2

## tensorflow
sudo rm -rf /usr/local/include
sudo ln -s /media/ephemeral0/anaconda/include /usr/local/include
sudo rm -rf /usr/local/lib
sudo ln -s /media/ephemeral0/anaconda/lib /usr/local/lib

sudo ln -s /opt/nvidia/cuda/ /usr/local/cuda
sudo cp /usr/local/include/cudnn.h /usr/local/cuda/include
sudo cp /usr/local/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# build conda package
conda build tensorflow --no-activate

# upload package to conda cloud so you'll go through this hell only once
anaconda upload /media/ephemeral0/anaconda/conda-bld/linux-64/tensorflow-0.9.0-py35_12.tar.bz2

conda install protobuf tensorflow --use-local --force

# if installing tensorflow from the anaconda repositories:
# then there is no need for bazel, JDK8, compiling tensorflow, etc.
conda install -c kevin-keraudren tensorflow

## Install latest Keras
pip install git+https://github.com/fchollet/keras.git --upgrade

# Do some cleaning
sudo yum clean
conda clean -all

## Install the necessary tools to survive in a terminal
sudo yum install emacs lynx htop

## using iterm and imgcat, we can look at images straight from the terminal
cd /media/ephemeral0/anaconda/bin/ && wget https://raw.githubusercontent.com/gnachman/iTerm2/master/tests/imgcat && chmod +x imgcat

# get the data (lynx is fast and reliable, scp is a shit hole) and unzip it
# https://www.kaggle.com/c/belkin-energy-disaggregation-competition/forums/t/5118/downloading-data-via-wget/112602
mkdir -p data && unzip train.zip -d data/train
