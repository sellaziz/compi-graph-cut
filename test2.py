import skimage
import matplotlib.pyplot as plt
import numpy as np
from main import *
from Img2Graph import image2graph

def test(show_im):
    ### CASE 1
    # img = np.zeros((10,10))
    # img[:4,:4] = 1
    # source = [(0,0), (0,1)] # Prior Object - Source S
    # sink = [(7,7)] # Prior Background - Sink T
    ### CASE 2
    ##CAT
    # img=getattr(skimage.data,'cat')()
    # source,sink=[(120,255)],[(200,400)]
    # img=skimage.color.rgb2gray(img)
    ## Horse
    img=getattr(skimage.data,'horse')()*1
    source,sink=[(120,100)],[(200,200)]
    ## Coins
    # img=getattr(skimage.data,'coins')()*1
    # img = img/255
    # source = [(a,b) for a in [50,120,200,265] for b in [45,110,150,223,270,350]]
    # sink = [(150,185)]
    if show_im:
        plt.imshow(img, cmap='gray')
        # Plot Source and Sink
        for pt in source:
            plt.scatter(pt[1],pt[0], color='r')
            plt.annotate(xy=(pt[1], pt[0]), text="Source", color='r')
        for pt in sink:
            plt.scatter(pt[1],pt[0], color='b')
            plt.annotate(xy=(pt[1], pt[0]), text="Sink",color='b')
        plt.show()
    print(f"Range of values [{np.min(img):0.2f}, {np.max(img):0.2f}], and shape : {img.shape}")
    G,A = image2graph(img, source, sink)
    P=grow(G, A)
    # print(P)
    orphans=augment(G,P)
    if show_im:
        plt.imshow(img, cmap='gray')
        # Plot Source and Sink
        for pt in source:
            plt.scatter(pt[1],pt[0], color='r')
            plt.annotate(xy=(pt[1], pt[0]), text="Source", color='r')
        for pt in sink:
            plt.scatter(pt[1],pt[0], color='b')
            plt.annotate(xy=(pt[1], pt[0]), text="Sink",color='b')
        # Plot path from growth stage
        # plt.plot([a[0] for a in P[1:-2]], [a[1] for a in P[1:-2]], 'r')
        plt.plot([a[1] for a in P[1:-2]], [a[0] for a in P[1:-2]], 'r')
        # for pt in P[1:-2]:
            # plt.scatter(pt[0], pt[1])
        # Plot orphans
        for pt in orphans:
            plt.scatter(pt[1], pt[0], color='darkorange')
            plt.annotate(xy=(pt[1], pt[0]), text="Orphan", color='darkorange')
        plt.show()
    # print(orphans)
    # img = np.zeros((3,3))
    # img[:2,:2] = 1
    # O = [(0,0), (0,1)] # Prior Object - Source S
    # B = [(2,2)] # Prior Background - Sink T
    # G,A = image2graph(img, O, B)

if __name__=="__main__":
    show_im=True
    test(show_im)