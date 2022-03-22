import numpy as np
from glob import glob
from imageio import imread, imwrite
import os
from skimage.color import rgb2gray
paths = [path for path in glob(os.path.join("results_bak","*.jpg"))]
masks = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in paths]
# bin_mask=[]
for mask in masks:
    #invert image
    B_pix=(mask < 0.5)
    W_pix=(mask >= 0.5)
    mask[B_pix] = 1
    mask[W_pix] = 0
    mask= 2*mask-1
# print(np.min(masks[0]),np.max(masks[0]), np.unique(masks[0]))
img_paths = [path for path in glob(os.path.join("dataset",'images',"*.jpg"))]
groundtruths = [np.array(rgb2gray(imread(path.replace('.jpg', '.png').replace('images','images-gt'), as_gray=False, pilmode="RGB")))*255 for path in img_paths]
# print(np.min(groundtruths[0]),np.max(groundtruths[0]), np.unique(groundtruths[0]))

def dice(gt, pred, with_contour=False):
    if with_contour:
        return _dice((gt>=128), (pred>=0))
    else:
        return _dice((gt>128), (pred>0))

def _dice(x,y):
    return 2 * np.sum((x==1)*(y==1))/ (np.sum(x==1) + np.sum(y==1))

import csv
with open('dices.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Image Basename", "DICE"])

    for i in range(len(masks)):
        # print(os.path.basename(paths[i])== os.path.basename(img_paths[i]))
        print(f"{dice(groundtruths[i], masks[i], False)*100:0.1f}")
        spamwriter.writerow([os.path.basename(paths[i]).replace('.jpg',''),str(f"{dice(groundtruths[i], masks[i], False)*100:0.1f}")])