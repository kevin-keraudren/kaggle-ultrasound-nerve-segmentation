import matplotlib
#matplotlib.use('cairo')
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

#import tensorflow as tf

import cv2
import itertools
from glob import glob
import time
from datetime import timedelta

from IPython.display import display
from IPython.display import SVG

import matplotlib.pyplot as plt
import sys
from base64 import b64encode

#import seaborn as sns
# scale font size
#sns.set(font_scale=1.8)

hastie_orange = (0.906,0.624,0)
hastie_blue = (0.337,0.706,0.914)
hastie_red = (0.678,0.137,0.137)
hastie_green = (0.114,0.412,0.078)

import os
# os.environ['OMP_NUM_THREADS'] = '8'
# os.environ['CC'] = "/usr/local/bin/clang-omp"
# os.environ['CXX'] = "/usr/local/bin/clang-omp++"

import theano
theano.config.exception_verbosity='high'
#theano.config.profile = 'True'
# theano.config.openmp = 'True'
# theano.config.cc = '/usr/local/bin/clang-omp'
# theano.config.cxx = '/usr/local/bin/clang-omp++'

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, AtrousConv2D, MaxPooling2D
from keras.utils.visualize_util import model_to_dot
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD
from keras.constraints import maxnorm, unitnorm

from keras import backend as K
import theano.tensor as T
from keras.optimizers import Optimizer
import keras.optimizers

from keras.layers.core import Layer
from keras import initializations

from keras.utils.visualize_util import plot
import scipy.ndimage as nd
from keras.callbacks import Callback, TensorBoard
from keras.regularizers import l1l2, activity_l1l2

EPOCH = 21
IMG0_NAME = "1_1"
#IMG0 = cv2.imread("cropped_positives/"+IMG0_NAME+".tif",0)
IMG0 = cv2.imread("cropped_positives_refinement/"+IMG0_NAME+".tif",0)
MEAN= 110.307
STD= 57.117

def save_model(model, model_name):
    directory = os.path.dirname(model_name)
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
    json_string = model.to_json()
    open(model_name+'_architecture.json', 'w').write(json_string)
    model.save_weights(model_name+'_weights.h5', overwrite=True)
    plot(model, to_file=model_name+'.png',show_shapes=True, show_layer_names=True)
    
class MyCallbackRegression(Callback):
    def on_batch_end(self, batch, logs={}):
        global IMG0
        global IMG0_NAME
        global EPOCH
        if (batch % 20) == 0:
            if not os.path.exists("callback_img0_regression"):
                os.makedirs("callback_img0_regression")
            EPOCH += 1
            test_regression(IMG0, self.model,
                                name="callback_img0_regression/batch_%s_%05d"%(IMG0_NAME,EPOCH))
        
    def on_epoch_end(self, epoch, logs={}):
        global IMG0
        global IMG0_NAME
        global EPOCH
        if not os.path.exists("callback_img0"):
            os.makedirs("callback_img0")
        EPOCH += 1
        test_regression(IMG0, self.model,
                            name="callback_img0_regression/epoch_%s_%05d"%(IMG0_NAME,EPOCH))
        
        if (epoch % 5) == 0:
            save_model(self.model,"callback_model_regression/model_epoch%05d"%epoch)
            
class MyCallbackClassification(Callback):
    def on_batch_end(self, batch, logs={}):
        global IMG0
        global IMG0_NAME
        global EPOCH
        if (batch % 20) == 0:
            if not os.path.exists("callback_img0"):
                os.makedirs("callback_img0")
            EPOCH += 1
            test_classification(IMG0, self.model,name="callback_img0/batch_%s_%05d"%(IMG0_NAME,EPOCH))
        
    def on_epoch_end(self, epoch, logs={}):
        global IMG0
        global IMG0_NAME
        global EPOCH
        if not os.path.exists("callback_img0"):
            os.makedirs("callback_img0")
        EPOCH += 1
        test_classification(IMG0, self.model, name="callback_img0/epoch_%s_%05d"%(IMG0_NAME,EPOCH))
        
        if (epoch % 5) == 0:
            save_model(self.model,"callback_model/model_epoch%05d"%epoch)
            
            
def test_regression(img, model, name):
    predictions = np.squeeze(model.predict(img[np.newaxis,...,np.newaxis]))
    shape = predictions.shape[:2]
    yx = np.rollaxis(np.mgrid[0:shape[0],0:shape[1]],0,3)
    predictions = yx + predictions
    predictions = np.round(predictions).astype('int')
    predictions = (predictions[:,:,0].flatten(),
                   predictions[:,:,1].flatten())
    
    flat_indices = np.ravel_multi_index(predictions, shape, mode='clip')
    bins = np.bincount(flat_indices, minlength=np.prod(shape))
    res = bins.reshape(shape)[...,np.newaxis].astype('float32')
    
    # artefacts of clipping
    res[0] = 0
    res[-1] = 0
    res[:,0] = 0
    res[:,-1] = 0

    print("max regression:",res.max())
    if res.max()>0:
        res *= 255/res.max()

    cv2.imwrite(name+"_regression.png",res.astype('uint8'))
    
def test_classification(img, model, name):
    seg = np.squeeze(model.predict(img[np.newaxis,...,np.newaxis]))
    print("range = [%s; %s]"%(seg.min(),seg.max()))
    
    seg[seg<0] = 0
    seg[seg>1] = 1
    
    print(name+"_classification.png")
    cv2.imwrite(name+"_classification_hard.png",(seg>0.5).astype('uint8')*255)
    cv2.imwrite(name+"_classification.png",(seg*255).astype('uint8'))
 
def predict( img, model_classification, model_regression, filename ):
    seg = np.squeeze(model_classification.predict(img[np.newaxis,...,np.newaxis]))
    
    predictions = np.squeeze(model_regression.predict(img[np.newaxis,...,np.newaxis]))
    norm = np.linalg.norm(predictions,axis=-1)
    shape = predictions.shape[:2]
    yx = np.rollaxis(np.mgrid[0:shape[0],0:shape[1]],0,3)
    predictions = yx + predictions
    predictions = np.round(predictions).astype('int')
    predictions = (predictions[:,:,0].flatten(),
                   predictions[:,:,1].flatten())
                   
    # threshold seg at 0.5
    seg[seg<0.5] = 0
    
    flat_indices = np.ravel_multi_index(predictions, shape, mode='clip')
    bins = np.bincount(flat_indices, minlength=np.prod(shape), weights=seg.flatten()*(1/(1+norm[...,np.newaxis])).flatten())
    res = bins.reshape(shape)[...,np.newaxis].astype('float32')
    
    # artefacts of clipping
    res[0] = 0
    res[-1] = 0
    res[:,0] = 0
    res[:,-1] = 0
    
    #res *= 1/(1+norm[...,np.newaxis])
    
    res = nd.gaussian_filter(res,0.5)
    _max = res.max()
    print("max:",_max)
    if _max>5:
        p = np.unravel_index(np.argmax(res),res.shape)
        mask = np.ones(res.shape)
        mask[p[0],p[1]] = 0
        distance = nd.distance_transform_edt(mask)
        res[distance>3]=0
    
        res = res.flat[flat_indices].reshape(shape)
        
        # take the intersection of the Hough back-projection
        # and the initial segmentation
        res = np.logical_and(res>0,seg>0)
        #res = nd.binary_closing(res,morphology.disk(3))
        res = nd.binary_fill_holes(res)
        
        # uncrop
        crop=16
        tmp_res = np.zeros((res.shape[0]+2*crop,res.shape[1]+2*crop),dtype='float32')
        tmp_res[crop:-crop,crop:-crop] = res
        
        # resize
        res = cv2.resize(tmp_res,(580,420),interpolation=cv2.INTER_CUBIC) > 0.5
        # if res.sum()<1000:
        #     res = np.zeros((420,580),dtype='uint8')
        #
    else:
        res = np.zeros((420,580),dtype='uint8')

    cv2.imwrite(filename+".png",res.astype('uint8')*255)
    
    return res
    
def dice_error(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred) 
    intersection = K.sum(y_true_f * y_pred_f)
    return (1.- (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon()))  
    
def dice_error_metric(y_true,y_pred):
    return dice_error(y_true, y_pred)   
       
def regression_error(y_true, y_pred):
    inside = K.cast(K.not_equal(y_true, 1e8), K.floatx())
    return K.sum(inside*K.square(y_pred - y_true),axis=[1,2,3])/K.sum(inside,axis=[1,2,3])
 
def regression_error_metric(y_true, y_pred):
    return K.mean(regression_error(y_true, y_pred))
    
def binary_crossentropy(y_true, y_pred):
    p1 =  0.175913638795
    p0=1-p1
    w = p1*K.cast(K.equal(y_true, 0), K.floatx()) + p0*K.cast(K.equal(y_true, 1), K.floatx())
    return K.sum(w*K.binary_crossentropy(y_pred, y_true),axis=[1,2,3])/K.sum(w,axis=[1,2,3])

def binary_crossentropy_metric(y_true, y_pred):
    return K.mean(binary_crossentropy(y_true, y_pred))
            
def get_model_regression(input_shape, filename=None,l1_value=1e-12, l2_value=1e-10):
    model = Sequential()
    
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
    border_mode='valid',input_shape=(input_shape[0],input_shape[1],1), dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AtrousConv2D(nb_filter=64, nb_row=3, nb_col=3, atrous_rate=(2, 2), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(AtrousConv2D(nb_filter=128, nb_row=3, nb_col=3, atrous_rate=(4, 4), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    
    model.add(AtrousConv2D(nb_filter=256, nb_row=3, nb_col=3, atrous_rate=(8, 8), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(nb_filter=256, nb_row=1, nb_col=1, border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(nb_filter=2, nb_row=1, nb_col=1, border_mode='valid', dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)
    ))
    
    model.add(Activation('linear'))

    print("compiling model")

    # load previously trained model
    if filename is not None:
        print("Loading weights from", filename)
        model.load_weights(filename)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[regression_error_metric],
    loss=regression_error)
    
    print("Model padding:",
          (input_shape[0] - model.output_shape[1])/2,
          (input_shape[1] - model.output_shape[2])/2)
    
    return model
    
def get_model_classification(input_shape, filename=None,l1_value=10e-12, l2_value=10e-14):
    model = Sequential()
    
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
    border_mode='valid',input_shape=(input_shape[0],input_shape[1],1), dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AtrousConv2D(nb_filter=64, nb_row=3, nb_col=3, atrous_rate=(2, 2), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(AtrousConv2D(nb_filter=128, nb_row=3, nb_col=3, atrous_rate=(4, 4), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(AtrousConv2D(nb_filter=512, nb_row=3, nb_col=3, atrous_rate=(8, 8), border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(nb_filter=512, nb_row=1, nb_col=1, border_mode='valid', dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(nb_filter=1, nb_row=1, nb_col=1, border_mode='valid', dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)
    ))
    
    model.add(Activation('sigmoid'))

    print("compiling model")

    # load previously trained model
    if filename is not None:
        print("Loading weights from", filename)
        model.load_weights(filename)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[dice_error_metric,binary_crossentropy_metric,'accuracy'],
    loss=binary_crossentropy)
    
    print("Model padding:",
          (input_shape[0] - model.output_shape[1])/2,
          (input_shape[1] - model.output_shape[2])/2)
    
    return model
    
def standardise(img):
    img -= MEAN
    img /+ STD
    return img
    
def get_training_data_regression(folder):
    X = []
    Y = []
    
    all_imgs = list(sorted(filter(lambda x: 'mask' not in x, glob(folder+"/*.tif"))))
    for f in all_imgs:
        img = standardise(cv2.imread(f, 0).astype('float32'))
        seg = (cv2.imread(f[:-len(".tif")]+"_mask.tif", 0)>0).astype('float32')
    
        center = np.array(nd.center_of_mass(seg),dtype='float32')
        yx = np.rollaxis(np.mgrid[0:seg.shape[0],0:seg.shape[1]],0,3).astype('float32')
        offsets = center - yx
    
        offsets[seg==0] = 1e8
    
        X.append(img[...,np.newaxis].astype('float32'))
        Y.append(offsets.astype('float32'))       
    
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')
    
def get_training_data_classification(folder):
    X = []
    Y = []
    
    all_imgs = list(sorted(filter(lambda x: 'mask' not in x, glob(folder+"/*.tif"))))
    for f in all_imgs:
        img = standardise(cv2.imread(f, 0).astype('float32'))
        seg = (cv2.imread(f[:-len(".tif")]+"_mask.tif", 0)>0).astype('float32')
    
        X.append(img[...,np.newaxis].astype('float32'))
        Y.append(seg[...,np.newaxis].astype('float32'))       
    
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')
    
def get_test_data(folder, downsample_factor=6):
    X = []
    X_names = []
    
    all_imgs = list(sorted(filter(lambda x: 'mask' not in x, glob(folder+"/*.tif"))))
    for f in all_imgs:
        img = standardise(cv2.imread(f, 0).astype('float32'))
        if downsample_factor > 1:
            img = cv2.resize(img,(img.shape[1]//downsample_factor, img.shape[0]//downsample_factor),
                             interpolation=cv2.INTER_AREA)
        X.append(img) 
        X_names.append(os.path.basename(f)[:-len('.tif')])
    
    return np.array(X, dtype='float32'), X_names
    
def get_model_classification_refinement(input_shape, filename=None,l1_value=10e-12, l2_value=10e-14,mode='valid'):
    
    def binary_crossentropy(y_true, y_pred):
        p1 =  0.412244897959
        p0=1-p1
        w = p1*K.cast(K.equal(y_true, 0), K.floatx()) + p0*K.cast(K.equal(y_true, 1), K.floatx())
        return K.sum(w*K.binary_crossentropy(y_pred, y_true),axis=[1,2,3])/K.sum(w,axis=[1,2,3])

    def binary_crossentropy_metric(y_true, y_pred):
        return K.mean(binary_crossentropy(y_true, y_pred))
        
    model = Sequential()
    
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
    border_mode=mode,input_shape=(input_shape[0],input_shape[1],1), dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
    border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(dim_ordering='tf'))
    
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3,
    border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3,
    border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(dim_ordering='tf'))

    model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3,
    border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3,
    border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(dim_ordering='tf'))    
    
    model.add(Convolution2D(nb_filter=512, nb_row=16, nb_col=16, border_mode=mode, dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=512, nb_row=1, nb_col=1, border_mode=mode, dim_ordering='tf',W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(nb_filter=1, nb_row=1, nb_col=1, border_mode=mode, dim_ordering='tf',
    W_regularizer=l1l2(l1=l1_value, l2=l2_value), activity_regularizer=activity_l1l2(l1=l1_value, l2=l2_value)
    ))
    
    model.add(Activation('sigmoid'))

    print("compiling model")

    # load previously trained model
    if filename is not None:
        print("Loading weights from", filename)
        model.load_weights(filename)

    model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[dice_error_metric,binary_crossentropy_metric,'accuracy'],
    loss=binary_crossentropy)
    
    print("Model padding:",
          (input_shape[0] - model.output_shape[1])/2,
          (input_shape[1] - model.output_shape[2])/2)
    
    return model
    
    