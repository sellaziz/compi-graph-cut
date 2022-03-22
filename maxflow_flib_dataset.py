'''
Maxflow using the library PyMaxflow, on the dataset
'''

import numpy as np
from imageio import imread, imwrite
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import maxflow
from examples_utils import plot_graph_2d
from Img2Graph import  image2graph_lib, initialize_priors
from skimage.transform import rescale, resize, downscale_local_mean
import os
from glob import glob

# Multiple images
# show_result=False
# nb_images=30

# One Image
show_result=True
nb_images=1

### Utils
def plot_img(imgs, nrow, ncol, figsize=10, legends=None, gray=True):
    fig, axs= plt.subplots(nrow, ncol, figsize=(figsize, figsize))
    n_tot=nrow*ncol
    for k in range(len(imgs)):
        idx=(k//ncol,k%ncol) if nrow>1 else k
        axs[idx].axis('off')
        if gray and len(imgs[k].shape)==2:
            axs[idx].imshow(imgs[k], cmap='gray')
        else:
            axs[idx].imshow(imgs[k])
        axs[idx].set_title(f"{legends[k]}")
    plt.show()


### Various images
# img = np.array(rgb2gray(imread("images/dog_small.jpg")))
# label = np.array(imread("images/dog_small_ST.jpg"))
# gt=np.zeros(label.shape)
# img = np.array(rgb2gray(imread("dataset/images/227092.jpg", as_gray=False, pilmode="RGB")))
# img_painted = np.array(imread("dataset/images-labels/227092-anno.png", as_gray=False, pilmode="RGB"))
# print(img_painted.shape)
# img = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
#                        anti_aliasing=True)
# source,sink=initialize_priors(img_painted)


# ### Resized Horse
# img = rgb2gray(imread("images/horse.jpg"))
# # img = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
# #                        anti_aliasing=True)
# img_painted = np.array(imread("images/horse_ST.jpg", as_gray=False, pilmode="RGB"))

# source,sink=[(img.shape[0]//2,img.shape[1]//2), (img.shape[0]//2+1,img.shape[1]//2+1)],[(2,2),(2+1,2+1)]
# source,sink=initialize_priors(img_painted)
### Image with squares
# img = np.zeros((10,10))
# img[:4,:4] = 1
# source = [(0,0)] # Prior Object - Source S
# sink = [(7,7)] # Prior Background - Sink T


# Using the dataset in the path : dataset/images
paths = [path for path in glob(os.path.join("dataset",'images',"*.jpg"))]
images = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in paths]
groundtruths = [np.array(rgb2gray(imread(path.replace('.jpg', '.png').replace('images','images-gt'), as_gray=False, pilmode="RGB"))) for path in paths]
labels = [np.array(imread(path.replace('.jpg', '-anno.png').replace('images','images-labels'), as_gray=False, pilmode="RGB")) for path in paths]

for i in range(nb_images):
    print(paths[i])
    fname=os.path.basename(paths[i])
    img=images[i]
    gt=groundtruths[i]
    label=labels[i]
    ## Plot Image
    if show_result:
        plot_img([img, gt, label], 1, 3, figsize=15, legends=["img","gt", "label"])
    # Initialize
    source,sink=initialize_priors(label, from_dataset=True, float_enc=False)
    # Do the min cut
    G, Rp, nodeids=image2graph_lib(img, source,sink,prior_as_index=True, nbins=10, σ=0.1, λ=0.1)
    G.maxflow()
    sgm = G.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    imwrite('results/'+fname, img2, as_gray=True, pilmode="RGB")

# Show the result.
if show_result:
    plt.imshow(img2)
    plt.show()

#### Get the DICE for the resulting images
# import numpy as np
# from glob import glob
# from imageio import imread, imwrite
# import os
# from skimage.color import rgb2gray
# import matplotlib.pyplot as plt

# paths = [path for path in glob(os.path.join("results_bak","*.jpg"))]
# masks = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in paths]
# # bin_mask=[]
# for mask in masks:
#     #invert image
#     B_pix=(mask < 0.5)
#     W_pix=(mask >= 0.5)
#     mask[B_pix] = 1
#     mask[W_pix] = 0
#     mask= 2*mask-1
# # print(np.min(masks[0]),np.max(masks[0]), np.unique(masks[0]))
# img_paths = [path for path in glob(os.path.join("dataset",'images',"*.jpg"))]
# groundtruths = [np.array(rgb2gray(imread(path.replace('.jpg', '.png').replace('images','images-gt'), as_gray=False, pilmode="RGB")))*255 for path in img_paths]
# # print(np.min(groundtruths[0]),np.max(groundtruths[0]), np.unique(groundtruths[0]))

# def dice(gt, pred, with_contour=False):
#     if with_contour:
#         return _dice((gt>=128), (pred>=0))
#     else:
#         return _dice((gt>128), (pred>0))

# def _dice(x,y):
#     return 2 * np.sum((x==1)*(y==1))/ (np.sum(x==1) + np.sum(y==1))

# import csv
# dices=[]
# bnames=[]
# with open('dices.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(["Image Basename", "DICE"])

#     for i in range(len(masks)):
#         # print(os.path.basename(paths[i])== os.path.basename(img_paths[i]))
#         print(f"{dice(groundtruths[i], masks[i], False)*100:0.1f}")
#         dices.append(dice(groundtruths[i], masks[i], False)*100)
#         bnames.append(os.path.basename(paths[i]).replace('.jpg',''))
#         spamwriter.writerow([os.path.basename(paths[i]).replace('.jpg',''),str(f"{dice(groundtruths[i], masks[i], False)*100:0.1f}")])

# plt.scatter(bnames,dices)
# plt.title("Dices")
# plt.xticks(range(len(bnames)), bnames, rotation=90)
# plt.show()