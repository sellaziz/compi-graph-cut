from PIL.Image import open
from main import *
from Img2Graph import *
import matplotlib.pyplot as plt

# Create a simple image
# img = np.zeros((3,3))
# img[:2,:2] = 1

# # Prior
# O = [(0,0)] # Prior Object - Source S
# B = [(2,2)] # Prior Background - Sink T
# priors = (O,B)

WIDTH, HEIGHT = 50,50
img = np.array(open('images\dog.jpg').convert('L').resize((WIDTH, HEIGHT)))/255
img_painted = np.array(open('images\dog_ST.jpg').resize((WIDTH, HEIGHT)))
priors = initialize_priors(img_painted)

G,_ = image2graph(img, *priors, nbins=10)

# G_sub = G.subgraph([n for n in G.nodes() if n not in ['T','S']])
# G_sub = G.edge_subgraph([e for e in list(G.edges) if (e[0]=='S' or e[1]=='S')])

# plt.figure(figsize=(10,10))
# pos=nx.spring_layout(G_sub) # pos = nx.nx_agraph.graphviz_layout(G)
# nx.draw_networkx(G_sub, pos, node_size=1500)
# labels = {k:f'{v:.2f}'for (k,v) in nx.get_edge_attributes(G_sub,'capacity').items()}
# nx.draw_networkx_edge_labels(G_sub,pos,edge_labels=labels)
# plt.show()

seg = graph2img(segment(G), *img.shape)

plt.figure(figsize=(15,7))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(seg, cmap='gray')
plt.show()
