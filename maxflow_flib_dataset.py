
import numpy as np
from imageio import imread
from matplotlib import pyplot as ppl
from skimage.color import rgb2gray
import maxflow
from examples_utils import plot_graph_2d
from Img2Graph import  image2graph_lib, initialize_priors
from skimage.transform import rescale, resize, downscale_local_mean


### Resized Horse
img = np.array(rgb2gray(imread("images/dog_small.jpg")))
img_painted = np.array(imread("images/dog_small_ST.jpg"))
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
source,sink=initialize_priors(img_painted)
### Image with squares
# img = np.zeros((10,10))
# img[:4,:4] = 1
# source = [(0,0)] # Prior Object - Source S
# sink = [(7,7)] # Prior Background - Sink T

## Plot Image
ppl.imshow(img, cmap='gray')
ppl.show()

G, Rp, nodeids=image2graph_lib(img, source,sink,prior_as_index=True, nbins=10, σ=0.1, λ=0.1)
# plot_graph_2d(G, nodeids.shape, plot_terminals=True)
G.maxflow()
sgm = G.get_grid_segments(nodeids)
print(sgm)
img2 = np.int_(np.logical_not(sgm))

# images = [open(path).convert('L') for path in glob(os.path.join("dataset",'images',"*.jpg"))]
# groundtruths = [open(path).convert('L') for path in glob(os.path.join("dataset",'images-gt',"*.png"))]
# labels = [open(path) for path in glob(os.path.join("dataset",'images-labels',"*.png"))]

# Show the result.
ppl.imshow(img2)
ppl.show()