import cv2
from glob import glob
import numpy as np

def imsave(img, filename):
    img = img.astype('float32')
    img -= img.min()
    img /= img.max()
    img *= 255
    cv2.imwrite(filename,img)

all_imgs = list(filter(lambda x: 'mask' not in x, glob("data/train/*.tif"))) #[:100]
all_imgs += list(filter(lambda x: 'mask' not in x, glob("data/test/*.tif")))

average = cv2.imread(all_imgs[0], 0).astype('float32')

for f in all_imgs[1:]:
    average += cv2.imread(f, 0).astype('float32')
    #average += cv2.imread(f, 0).astype('float32')[:,::-1]
    
average /= 2*len(all_imgs)

np.save("average.npy",average)

std=0
for f in all_imgs:
    img = cv2.imread(f, 0).astype('float32')
    std += (img-average)*(img-average)
    #std += (img-average[:,::-1])*(img-average[:,::-1])

std /= 2*len(all_imgs)

print("STD", np.sqrt(np.mean(std)))

np.save("std.npy",np.sqrt(std))

imsave(np.sqrt(std),"std.png")
imsave(average,"average.png")