
import numpy as np
from imageio import imread, imwrite
from matplotlib import pyplot as ppl
from skimage.color import rgb2gray
import maxflow
from examples_utils import plot_graph_2d
from Img2Graph import capacity, Rp, dist, Bpq, prior_probs, initialize_priors 
from skimage.transform import rescale, resize, downscale_local_mean
import os
from glob import glob

#### CREATE ./results directory before running

### Resized Horse
# img = np.array(rgb2gray(imread("images/dog.jpg")))
# img_painted = np.array(imread("images/dog_ST.jpg"))
# img = np.array(rgb2gray(imread("dataset/images/227092.jpg", as_gray=False, pilmode="RGB")))
# img_painted = np.array(imread("dataset/images-labels/227092-anno.png", as_gray=False, pilmode="RGB"))
# # print(img_painted.shape)
# # img = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
# #                        anti_aliasing=True)
# source,sink=initialize_priors(img_painted)


# ### Resized Horse
# img = rgb2gray(imread("images/horse.jpg"))
# # img = resize(img, (img.shape[0] // 4, img.shape[1] // 4),
# #                        anti_aliasing=True)
# source,sink=[(img.shape[0]//2,img.shape[1]//2)],[(2,2)]

### Image with squares
# img = np.zeros((10,10))
# img[:4,:4] = 1
# source = [(0,0)] # Prior Object - Source S
# sink = [(7,7)] # Prior Background - Sink T

## Plot Image
# ppl.imshow(img, cmap='gray')
# ppl.show()

def image2graph(img, O, B, nbins=10, alpha=10):
    """Convert the input image into a graph for the segmentation part.
        O, B: Object and Background pixels as list of tuple.
    """
    n,p = img.shape
    # print(n, p)
    # print(tuple(np.array(O).T))
    # probs = prior_probs(img[O], img[B], nbins, alpha)
    probs = prior_probs(img[tuple(np.array(O).T)], img[tuple(np.array(B).T)], nbins, alpha)

    # Initialize graph without S and T nodes
    G = maxflow.Graph[float](0, 0)
    nodeids = G.add_grid_nodes(img.shape)
    for i in range(n):
        for j in range(p):
            if i+1 < n:
                G.add_edge(nodeids[i,j],nodeids[i+1,j], capacity(img,(i,j),(i+1,j), O,B,probs),capacity(img,(i,j),(i+1,j), O,B, probs))
            if j+1 < p:
                G.add_edge(nodeids[i,j],nodeids[i,j+1], capacity(img,(i,j),(i,j+1), O,B,probs),capacity(img,(i,j),(i,j+1), O,B, probs))
    
    # Compute K
    K = 0
    Gnx=G.get_nx_graph()
    for x in Gnx.nodes:
        K = max(K, np.sum([Gnx.get_edge_data(x,y)['weight'] for y in Gnx.neighbors(x)]))
    K += 1
    print(f"K = {K}")

    # Add edges for S and T
    for i in range(n):
        for j in range(p):
            G.add_tedge(nodeids[i,j], capacity(img,(i,j),'S',O,B,probs, K=K),capacity(img,(i,j),'T',O,B,probs,K=K))
    
    # # Initial active nodes
    print('graph init done')
    # plot_graph_2d(G, nodeids.shape, plot_terminals=True)
    return G, nodeids

# img_paths=[path for path in glob(os.path.join("dataset",'images',"*.jpg"))]
# images = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in img_paths]
# groundtruths = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in glob(os.path.join("dataset",'images-gt',"*.png"))]
# labels = [np.array(imread(path, as_gray=False, pilmode="RGB")) for path in glob(os.path.join("dataset",'images-labels',"*.png"))]

paths = [path for path in glob(os.path.join("dataset",'images',"*.jpg"))]
images = [np.array(rgb2gray(imread(path, as_gray=False, pilmode="RGB"))) for path in paths]
groundtruths = [np.array(rgb2gray(imread(path.replace('.jpg', '.png').replace('images','images-gt'), as_gray=False, pilmode="RGB"))) for path in paths]
labels = [np.array(imread(path.replace('.jpg', '-anno.png').replace('images','images-labels'), as_gray=False, pilmode="RGB")) for path in paths]

for i in range(30):
    #### CREATE ./results directory before running
    print(paths[i])
    fname=os.path.basename(paths[i])
    img=images[i]
    # ppl.imshow(img, cmap='gray')
    # ppl.show()
    gt=groundtruths[i]
    label=labels[i]
    source,sink=initialize_priors(label)
    # ppl.plot(source)
    # ppl.show()
    G, nodeids=image2graph(img, source,sink, nbins=10)
    G.maxflow()
    sgm = G.get_grid_segments(nodeids)
    img2 = np.int_(np.logical_not(sgm))
    imwrite('results/'+fname, img2, as_gray=True, pilmode="RGB")
# Show the result.
# ppl.imshow(img2)
# ppl.show()