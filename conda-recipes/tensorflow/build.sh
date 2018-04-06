#!/bin/bash

# The first error that popped up was zlib not being found.
# It could be solved by the bazel build option `--copt "-I$PREFIX/include/"`.
# However, it was followed by the following error:
#
# this rule is missing dependency declarations for the following files included by
# 'external/grpc/src/core/lib/compression/message_compress.c':
#  '/home/ec2-user/anaconda/envs/_build/include/zlib.h'
#  '/home/ec2-user/anaconda/envs/_build/include/zconf.h'.
#
# This second error could potentially be fixed by 'cxx_builtin_include_directory',
# but in the end what worked well was just symlinking the hell out of it:
#
# sudo rm -rf /usr/local/include
# sudo ln -s /media/ephemeral0/anaconda/include /usr/local/include
# sudo rm -rf /usr/local/lib
# sudo ln -s /media/ephemeral0/anaconda/lib /usr/local/lib
#
# Symlinks can similarly be used to go from the NVIDIA install of cuda and
# the Anaconda install of cudnn to what tensorflow expects by default:
#
# sudo ln -s /opt/nvidia/cuda/ /usr/local/cuda
# sudo cp /usr/local/include/cudnn.h /usr/local/cuda/include
# sudo cp /usr/local/lib/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# Set up symlinks to python include normally performed by ./configure
export PYTHON_BIN_PATH=$PREFIX/bin/python
(./util/python/python_config.sh --setup "$PYTHON_BIN_PATH";) || exit -1

# Use conda installed swig
# http://stackoverflow.com/questions/33885276
echo "#!/bin/bash" > tensorflow/tools/swig/swig.sh
echo "`which swig` \"\$@\"" >> tensorflow/tools/swig/swig.sh
cat tensorflow/tools/swig/swig.sh

# build wheel using bazel
bazel clean --expunge

echo "Answers: N y /usr/bin/gcc 7.5 /usr/local/cuda 4 /usr/local/cuda 3.0"
# the configuration is interactive!
./configure
bazel build --config=cuda -c opt //tensorflow/tools/pip_package:build_pip_package
mkdir $SRC_DIR/tensorflow_pkg
bazel-bin/tensorflow/tools/pip_package/build_pip_package $SRC_DIR/tensorflow_pkg

# install using pip from the whl file
pip install --no-deps $SRC_DIR/tensorflow_pkg/*.whl
