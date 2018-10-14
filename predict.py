from model import *


def predict_classification(img, model, filename):
    seg = np.squeeze(model.predict(img[np.newaxis,...,np.newaxis]))
    print("range = [%s; %s]"%(seg.min(),seg.max()))
    # resize
    seg = cv2.resize(seg,(580,420),interpolation=cv2.INTER_CUBIC)
    seg[seg<0] = 0
    seg[seg>1] = 1
    seg = (seg > 0.5).astype('uint8')*255
    cv2.imwrite(filename,seg.astype('uint8'))


def predict( img, model_classification, model_regression, filename, padding=32 ):
    
    if padding > 0:
        pad_width = ((padding, padding),
                     (padding, padding))
        img = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    
    seg = np.squeeze(model_classification.predict(img[np.newaxis,...,np.newaxis]))
    print("range = [%s; %s]"%(seg.min(),seg.max()))
    cv2.imwrite(filename+"_seg.png",(seg>0.5).astype('uint8')*255)
    
    predictions = np.squeeze(model_regression.predict(img[np.newaxis,...,np.newaxis]))
    shape = predictions.shape[:2]
    yx = np.rollaxis(np.mgrid[0:shape[0],0:shape[1]],0,3)
    predictions = yx + predictions
    predictions = np.round(predictions).astype('int')
    predictions = (predictions[:,:,0].flatten(),
                   predictions[:,:,1].flatten())
                   
    # threshold seg at 0.5
    seg[seg<0.5] = 0
    
    flat_indices = np.ravel_multi_index(predictions, shape, mode='clip')
    bins = np.bincount(flat_indices, minlength=np.prod(shape), weights=seg.flatten())
    res = bins.reshape(shape)[...,np.newaxis].astype('float32')
    
    # artefacts of clipping
    res[0] = 0
    res[-1] = 0
    res[:,0] = 0
    res[:,-1] = 0
    
    res = nd.gaussian_filter(res,1.0)
    cv2.imwrite(filename+"_votes.png",(res/res.max()*255).astype('uint8'))
    _max = res.max()
    print("max:",_max, filename)
    if _max>25: # value optimised on validation dataset
        p = np.unravel_index(np.argmax(res),res.shape)
        mask = np.ones(res.shape)
        mask[p[0],p[1]] = 0
        distance = nd.distance_transform_edt(mask)
        res[distance>10]=0
        res = res.flat[flat_indices].reshape(shape)
    
        # take the intersection of the Hough back-projection
        # and the initial segmentation
        res = np.logical_and(res>0,seg>0)
        res = nd.binary_fill_holes(res)
        
        if padding > 0:
            res = res[padding:-padding,padding:-padding]
        
        # resize
        res = cv2.resize(res.astype('float32'),(580,420),interpolation=cv2.INTER_CUBIC) > 0.5
    else:
        res = np.zeros((420,580),dtype='uint8')

    cv2.imwrite(filename+"_prediction.png",res.astype('uint8')*255)


input_shape=(274, 354, 1)
downsample_factor = 2
regression_model = get_model(input_shape,task='regression',training=False,
                                 filename="saved_models/regression/model_epoch00055_weights.h5")
classification_model = get_model(input_shape,task='classification',training=False,
                                 filename="saved_models/classification/model_epoch00055_weights.h5")
                                 
keras_plot(regression_model, to_file='regression_model.png', show_shapes=True, show_layer_names=False)
keras_plot(classification_model, to_file='classification_model.png', show_shapes=True, show_layer_names=False)

os.makedirs('prediction_test',exist_ok=True)

print("predicting testing data")
X_test_names = list( glob("data/test/*.tif"))
for i in range(len(X_test_names)):
    img = standardise(cv2.imread(X_test_names[i], 0).astype('float32'))
    if downsample_factor > 1:
        img = cv2.resize(img,(img.shape[1]//downsample_factor, 
                         img.shape[0]//downsample_factor),interpolation=cv2.INTER_AREA)
    print(img.shape)
    name=os.path.basename(X_test_names[i])[:-len('.tif')]
    predict(img, classification_model,regression_model, 
            filename="prediction_test/%s"%name)
    #predict_classification(img, classification_model,
    #        filename="prediction_test/%s_prediction.png"%name)        
            
            