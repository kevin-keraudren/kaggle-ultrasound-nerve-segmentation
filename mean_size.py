import cv2
from glob import glob
import numpy as np

all_imgs = list(filter(lambda x: 'mask' in x, glob("data/train/*.tif"))) 
m = 0
n = 0
_min = np.inf
for f in all_imgs:
    s = (cv2.imread(f,0)>0).astype('float').sum()
    if s > 0:
        m += s
        n += 1
        if s < _min:
            _min = s
        
print("mean:",m/n)
print("min:",_min)