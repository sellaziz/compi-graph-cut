import matplotlib.pyplot as plt
import numpy as np
from main import *
from Img2Graph import graph2img
from matplotlib.animation import FuncAnimation

def Simple_Image(image, cmap, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(-.5, 3, 1)
    minor_ticks = np.arange(-.5, 3, .5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.1)
    ax.grid(which='major', alpha=0.7)

    ax.imshow(image, cmap=cmap)
    ax.scatter([0],[0], c='red', s=30)
    ax.scatter([2],[2], c='blue', s=30)
    ax.text(.03,-.07,'O', c='red')
    ax.text(2.03,1.93,'B', c='blue')
    plt.show()

def visualize_Rp(Rp, N=100):
    X = np.linspace(0, 1 + 1/N, N)
    plt.figure(figsize=(15,5))

    plt.subplot(121)
    plt.title("Regional term Oject")
    plt.plot(X, Rp["obj"](X), c='r')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('$R_{p}(obj)$')

    plt.subplot(122)
    plt.title("Regional term Background")
    plt.plot(X, Rp["bkg"](X), c='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('$R_{p}(bkg)$')
    
    plt.show()

def segmentation_images(G, img):
    A = ['S','T']
    images = []
    while True:

        P = grow(G, A)
        if not P:
            break
        Orphans = augment(G, P)
        A = adopt(G, Orphans, A)
        images.append(graph2img(G, *img.shape))
    
    images.append(graph2img(G, *img.shape))
    return G, images

def segmentation_animation(images, interval=1):
    
    fig = plt.figure()
    plt.title('Segementation process')
    im = plt.imshow(images[0], cmap='seismic')

    def animate(i):
        return [im.set_array(images[i])]
    
    frames = [i for i in range(0, len(images), interval)] + [len(images)-1]
    anim_created = FuncAnimation(fig, animate, frames=iter(frames))
    
    video = anim_created.to_jshtml()
    return video