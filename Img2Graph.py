import networkx as nx
import numpy as np
import scipy.stats

#############################################################
def initialize_priors(img_painted):
    """Image painted with red for Object and blue for Source.
        Return the Object and Background pixels."""
    red, green, blue = np.transpose(img_painted, (2,0,1))
    O_mask = (red > 200) * (green < 100) * (blue < 100)
    B_mask = (red < 80) * (green < 80) * (blue > 140)
    O = [tuple(idx) for idx in np.argwhere(O_mask)]
    B = [tuple(idx) for idx in np.argwhere(B_mask)]
    return O, B

#############################################################
def graph2img(G, n, p):
    seg = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            if G.nodes[(i,j)]['tree'] == 'S':
                seg[i,j] = 1
            elif G.nodes[(i,j)]['tree'] == 'T':
                seg[i,j] = -1
    return seg

#############################################################
def prior_probs(O_vals, B_vals, nbins, alpha:int=10):
    # Add data all along to prevent probability of 0
    data_O = np.hstack((np.repeat(O_vals,alpha), np.linspace(0,1+1/nbins, nbins)))
    data_B = np.hstack((np.repeat(B_vals,alpha), np.linspace(0,1+1/nbins, nbins)))

    hist_O = np.histogram(data_O, bins=nbins, density=True)
    hist_B = np.histogram(data_B, bins=nbins, density=True)

    hist_dist_O = scipy.stats.rv_histogram(hist_O)
    hist_dist_B = scipy.stats.rv_histogram(hist_B)

    proba_O = lambda x : hist_dist_O.pdf(x) / nbins
    proba_B = lambda x : hist_dist_B.pdf(x) / nbins
    
    return dict(obj=proba_O, bkg=proba_B)

def dist(p, q):
    """Distance L1 between pixels p and q."""
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def Bpq(img, p, q, sig=1):
    """The boundary properties term between pixels p and q."""
    return np.exp(-(img[p]-img[q])**2 / (2*sig**2)) / dist(p,q)

def Rp(img, p, probs, label):
        """The regional term for a pixel p, with eps to prevent infinity."""
        return -np.log(probs[label](img[p]))

def capacity(img, p, q, O, B, probs, λ=1, K=None):
    """Compute weight between nodes p and q.
    When q is the source S or the sink T, K must be given.
    """
    if q == 'S':
        if p in O:
            return K
        if p in B:
            return 0
        return λ*Rp(img,p,probs,"bkg")
    
    if q == 'T':
        if p in O:
            return 0
        if p in B:
            return K
        return λ*Rp(img,p,probs,"obj")
    
    return Bpq(img,p,q)
    
#############################################################
def image2graph(img, O, B, nbins=10, alpha=10, prior_as_index=False):
    """Convert the input image into a graph for the segmentation part.
        O, B: Object and Background pixels as list of tuple.
    """

    n,p = img.shape
    if prior_as_index:
        probs = prior_probs(img[tuple(np.array(O).T)], img[tuple(np.array(B).T)], nbins, alpha)
    else:
        probs = prior_probs(img[O].flatten(), img[B].flatten(), nbins, alpha)

    # Initialize graph without S and T nodes
    G = nx.Graph()
    for i in range(n):
        for j in range(p):
            if i+1 < n:
                G.add_edge((i,j),(i+1,j), capacity=capacity(img, (i,j), (i+1,j), O, B, probs))
            if j+1 < p:
                G.add_edge((i,j),(i,j+1), capacity=capacity(img, (i,j), (i,j+1), O, B, probs))
    
    # Compute K
    K = 0
    for x in G.nodes:
        K = max(K, np.sum([G.get_edge_data(x,y)['capacity'] for y in G.neighbors(x)]))
    K += 1

    # Add edges for S and T
    for i in range(n):
        for j in range(p):
            G.add_edge((i,j),'S', capacity=capacity(img, (i,j), 'S', O, B, probs, K=K))
            G.add_edge((i,j),'T', capacity=capacity(img, (i,j), 'T', O, B, probs, K=K))
    
    # Add other attributes
    nx.set_node_attributes(G, None, "parent")
    nx.set_node_attributes(G, None, "tree")
    nx.set_edge_attributes(G, 0, "flow")

    # Set S & T trees
    G.nodes['S']['tree'] = 'S'
    G.nodes['T']['tree'] = 'T'

    return G, probs
