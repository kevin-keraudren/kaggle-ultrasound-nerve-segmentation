"""
https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/submission.py

"""
import numpy as np
import cv2
from glob import glob
import os

#from sklearn.cluster import MeanShift

def postprocess(img):
    # filter by size (mean size: 7126; min size: 2684)
    if img.sum() < 1000:
        return np.zeros(img.shape)
        
    # # Mean-shift clustering and remove small clusters
    # pts = np.argwhere(img)[::8]
    # ms = MeanShift(bandwidth=50, n_jobs=1)
    # ms.fit(pts)
    #
    # n_labels = len(ms.cluster_centers_)
    # colors = np.random.randint(0,255,(n_labels,3))
    # labels = ms.labels_
    # res = np.zeros((*img.shape,3),dtype='uint8')
    # res[pts[:,0],pts[:,1]] = colors[labels]
    # cv2.imshow("ms",res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return img

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])
 
 
first_row = 'img,pixels'
file_name = 'submission.csv'
f = open(file_name,'w')
f.write(first_row + '\n')

# all_predictions = list(filter(lambda x: 'proba' not in x, glob("prediction_train/*.png")))
# all_ids = list(map(lambda x: list(map(int,os.path.basename(x)[:-len('_prediction.png')].split('_'))),
#                    all_predictions))
# all_ids = np.array(all_ids,dtype='int')
# indices = np.lexsort((all_ids[:,1],all_ids[:,0]))

all_predictions = list(glob("prediction_test_combined/*.png"))
all_ids = list(map(lambda x: int(os.path.basename(x)[:-len('_prediction.png')]),
                    all_predictions))
indices = np.argsort(all_ids)

s1 = 0
s0 = 0
for i in indices:
    labels = cv2.imread(all_predictions[i],0)>0
    s1 += np.sum(labels==1)
    s0 += np.sum(labels==0)
    #labels = postprocess(labels)
    rle = run_length_enc(labels)
    #f.write(str(all_ids[i,0])+','+str(all_ids[i,1])+' '+rle+'\n')
    f.write(str(all_ids[i])+','+rle+'\n')
    
f.close()

print("p1 = ", s1/(s0+s1))