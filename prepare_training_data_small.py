"""
Output:

Number of non-empty segmentations: 2323/5635
x_size:  91.43650452 25.8574970283
y_size:  116.828239346 18.8273583405
p1 =  0.27742949244

"""

import os
import cv2
from glob import glob
import numpy as np
import scipy.ndimage as nd
    
def bbox( img, crop=False ):
    """
    Bounding box.
    """
    pts = np.transpose(np.nonzero(img>0))
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    if not crop:
        return ( x_min, y_min, x_max, y_max )
    else:
        return img[y_min:y_max+1,
                   x_min:x_max+1]

output_dir = "cropped_positives"
os.makedirs(output_dir, exist_ok=True)

all_segs = list(filter(lambda x: 'mask' in x, glob("data/train/*.tif")))

crop = 15
downsample_factor = 6

crop *= downsample_factor

shape_seg = (200,200)
shape_img = (shape_seg[0]+2*crop, shape_seg[1]+2*crop)

average_img = np.zeros(np.array(shape_img)//downsample_factor,dtype='float32')
average_seg = np.zeros(np.array(shape_seg)//downsample_factor,dtype='float32')
n = 0
s1 = 0
s0 = 0

all_x_size=[]
all_y_size=[]
pad_size = 180
for f_seg in all_segs:
    seg = cv2.imread(f_seg,0)
    if seg.sum() == 0:
        continue
    f_img = f_seg[:-len("_mask.tif")]+".tif"
    img = cv2.imread(f_img, 0)
    center = np.round(nd.center_of_mass(seg)).astype('int')
    x_min, y_min, x_max, y_max = bbox(seg)
    all_x_size.append(x_max-x_min)
    all_y_size.append(y_max-y_min)
    
    img = np.pad(img, [(pad_size, pad_size)] * 2, mode='edge')
    seg = np.pad(seg, [(pad_size, pad_size)] * 2, mode='constant')
    img = img[pad_size+center[0]-shape_img[0]//2:pad_size+center[0]+shape_img[0]//2+1,
              pad_size+center[1]-shape_img[1]//2:pad_size+center[1]+shape_img[1]//2+1]
    seg = seg[pad_size+center[0]-shape_seg[0]//2:pad_size+center[0]+shape_seg[0]//2+1,
              pad_size+center[1]-shape_seg[1]//2:pad_size+center[1]+shape_seg[1]//2+1]
              
    if downsample_factor > 1:
        img = cv2.resize(img,(img.shape[1]//downsample_factor, img.shape[0]//downsample_factor),
                         interpolation=cv2.INTER_AREA)
        seg = cv2.resize(seg,(seg.shape[1]//downsample_factor, seg.shape[0]//downsample_factor),
                         interpolation=cv2.INTER_NEAREST)
                       
    average_img += img
    average_seg += seg
                         
    # center = np.round(nd.center_of_mass(seg)).astype('int')
    # mask = np.ones(seg.shape)
    # mask[center[0],center[1]] = 0
    # distance = nd.distance_transform_edt(mask)
    # seg = (distance < 3).astype('uint8')*255
    
        
    cv2.imwrite(os.path.join(output_dir,os.path.basename(f_seg)),seg)
    cv2.imwrite(os.path.join(output_dir,os.path.basename(f_img)),img)

    n +=1
    s1 += np.sum(seg>0)
    s0 += np.sum(seg==0)
    
all_x_size = np.array(all_x_size)
all_y_size = np.array(all_y_size)
    
print("Number of non-empty segmentations: %s/%s" % (n,len(all_segs)))

print("x_size: ", all_x_size.mean(), all_x_size.std())
print("y_size: ", all_y_size.mean(), all_y_size.std())

def make_average(average, n):
    average /= n
    average -= average.min()
    average /= average.max()
    average *= 255
    return average
    
average_img = make_average(average_img,n)
average_seg = make_average(average_seg,n)

cv2.imwrite("average_img.png", average_img)
cv2.imwrite("average_seg.png", average_seg)

print("p1 = ", s1/(s0+s1))

all_cropped_imgs = list(filter(lambda x: 'mask' not in x, glob(output_dir+"/*.tif")))
data = []
for f in all_cropped_imgs:
    data.append(cv2.imread(f).astype("float32"))
    
data = np.array(data)
print("MEAN = ", data.mean())
print("STD = ", data.std())