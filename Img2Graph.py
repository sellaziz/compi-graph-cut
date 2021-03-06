# Functions to go from an image to a graph and vice-versa

import networkx as nx
import numpy as np
import scipy.stats
from scipy.stats import gaussian_kde
import maxflow

#############################################################
def initialize_priors(img_painted, from_dataset=False, float_enc=False):
    """Image painted with red for Object and blue for Source.
        Return the Object and Background pixels."""
    if float_enc:
        red, green, blue = np.transpose(img_painted, (2,0,1))*255
    else:
        red, green, blue = np.transpose(img_painted, (2,0,1))
    if from_dataset:
        O_mask = (red > 50) * (green < 70) * (blue < 70)
        B_mask = (red > 70) * (green > 70) * (blue > 70) # white mask
    else:
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

def graph2cuts(G, n, p):
    segH = np.zeros((n,p))
    for i in range(n-1):
        for j in range(p):
            edge = G.get_edge_data((i,j), (i+1,j))
            if edge['capacity'] > edge['flow']:
                segH[i,j] = 1
            else:
                segH[i,j] = -1
    
    segW = np.zeros((n,p))
    for i in range(n):
        for j in range(p-1):
            edge = G.get_edge_data((i,j), (i,j+1))
            if edge['capacity'] > edge['flow']:
                segW[i,j] = 1
            else:
                segW[i,j] = -1
    
    return segH, segW

#############################################################

def Rp_prob(O_vals, B_vals):
    if isinstance(B_vals[0], np.ndarray) or len(set(list(B_vals))) > 2:
        log_pdf_O = gaussian_kde(O_vals.T).logpdf
        Rp_O = lambda x : np.clip(np.log(len(O_vals))-log_pdf_O(x.T), a_min=0, a_max=None)
    else:
        log_pdf_O = gaussian_kde(list(np.linspace(0,1,10)) + [O_vals[0]], weights=[1]*10+[100]).logpdf
        Rp_O = lambda x : np.clip(-log_pdf_O(x.T), a_min=0, a_max=None)

    if isinstance(B_vals[0], np.ndarray) or len(set(list(B_vals))) > 2:
        log_pdf_B = gaussian_kde(B_vals.T).logpdf
        Rp_B = lambda x : np.clip(np.log(len(B_vals))-log_pdf_B(x.T), a_min=0, a_max=None)
    else:
        log_pdf_B = gaussian_kde(list(np.linspace(0,1,10)) + [B_vals[0]], weights=[1]*10+[100]).logpdf
        Rp_B = lambda x : np.clip(-log_pdf_B(x.T), a_min=0, a_max=None)

    return dict(obj=Rp_O, bkg=Rp_B)

def create_edges(n,p):
    I,J = np.meshgrid(np.arange(n), np.arange(p))
    coord = np.vstack([I.ravel(), J.ravel()]).T
    coord_I = coord[coord[:,0] < n-1]
    coord_J = coord[coord[:,1] < p-1]
    edge_I = np.hstack((coord_I, coord_I + [1,0])).reshape((n-1)*p,2,2)
    edge_J = np.hstack((coord_J, coord_J + [0,1])).reshape(n*(p-1),2,2)
    return np.vstack((edge_I,edge_J))

#############################################################
def image2graph(img, O, B, prior_as_index=False, ??=1, ??=1, **kwargs):
    """Convert the input image into a graph for the segmentation part.
        O, B: Object and Background pixels as list of tuple.
    """
    assert len(O) != 0 and len(B) != 0,"O or B is empty!"
    n,p = img.shape[:2]
    
    if prior_as_index:
        O_vals, B_vals = img[tuple(np.array(O).T)], img[tuple(np.array(B).T)]
    else:
        O_vals, B_vals = img[O], img[B]

    # Regional term Rp
    Rp = Rp_prob(O_vals, B_vals)

    # Initialize graph without S and T nodes
    edges = create_edges(n,p)
    edges_tup = list(map(lambda x : ((tuple(x[0]),tuple(x[1]))),edges))

    # Compute capacities (Bpq)
    if len(img.shape) > 2:
        capacities = np.exp(-np.sum(((img[tuple(edges[:,0].T)]-img[tuple(edges[:,1].T)])**2),axis=1) / (2*??**2))
    else:
        capacities = np.exp(-((img[tuple(edges[:,0].T)]-img[tuple(edges[:,1].T)])**2) / (2*??**2))
    
    # Add nodes
    G = nx.Graph(edges_tup)
    # Add capacities
    nx.set_edge_attributes(G, dict(map(lambda x,c : (x, {"capacity":c}), edges_tup, capacities)))
    
    # Compute K
    K = 1 + np.max([np.sum([G.get_edge_data(x,y)['capacity'] for y in G.neighbors(x)]) for x in G.nodes])

    # Add edges for S and T
    if prior_as_index:
        for i in range(n):
            for j in range(p):
                if (i,j) in B:
                    G.add_edge((i,j),'T', capacity=K)
                elif (i,j) in O:
                    G.add_edge((i,j),'S', capacity=K)
                else:
                    G.add_edge((i,j),'T', capacity=??*Rp["obj"](img[(i,j)])[0])
                    G.add_edge((i,j),'S', capacity=??*Rp["bkg"](img[(i,j)])[0])
    else:
        for i in range(n):
            for j in range(p):
                if B[i,j] == 1:
                    G.add_edge((i,j),'T', capacity=K)
                elif O[i,j] == 1:
                    G.add_edge((i,j),'S', capacity=K)
                else:
                    G.add_edge((i,j),'T', capacity=??*Rp["obj"](img[(i,j)])[0])
                    G.add_edge((i,j),'S', capacity=??*Rp["bkg"](img[(i,j)])[0])

    
    # Add other attributes
    nx.set_node_attributes(G, None, "parent")
    nx.set_node_attributes(G, None, "tree")
    nx.set_edge_attributes(G, 0, "flow")

    # Set S & T trees
    G.nodes['S']['tree'] = 'S'
    G.nodes['T']['tree'] = 'T'

    return G, Rp

def image2graph_lib(img, O, B, prior_as_index=False, ??=1, ??=1, **kwargs):
    """Convert the input image into a graph for the segmentation part.
        O, B: Object and Background pixels as list of tuple.
    """
    assert len(O) != 0 and len(B) != 0,"O or B is empty!"
    n,p = img.shape[:2]
    
    if prior_as_index:
        O_vals, B_vals = img[tuple(np.array(O).T)], img[tuple(np.array(B).T)]
    else:
        O_vals, B_vals = img[O], img[B]

    # Regional term Rp
    Rp = Rp_prob(O_vals, B_vals)

    # Initialize graph without S and T nodes
    edges = create_edges(n,p)
    edges_tup = list(map(lambda x : ((tuple(x[0]),tuple(x[1]))),edges))

    # Compute capacities (Bpq)
    if len(img.shape) > 2:
        capacities = np.exp(-np.sum(((img[tuple(edges[:,0].T)]-img[tuple(edges[:,1].T)])**2),axis=1) / (2*??**2))
    else:
        capacities = np.exp(-((img[tuple(edges[:,0].T)]-img[tuple(edges[:,1].T)])**2) / (2*??**2))
    
    # Add nodes
    G = maxflow.Graph[float](0, 0)
    nodeids = G.add_grid_nodes(img.shape)
    # Add capacities
    # nx.set_edge_attributes(G, dict(map(lambda x,c : (x, {"capacity":c}), edges_tup, capacities)))
    # map(lambda x,c : G.add_edge(nodeids[x[0][0],x[0][1]], nodeids[x[1][0],x[1][1]], c, c), edges_tup, capacities)
    for i in range(len(edges_tup)):
        x=edges_tup[i]
        c=capacities[i]
        G.add_edge(nodeids[x[0][0],x[0][1]], nodeids[x[1][0],x[1][1]], c, c)
    # Compute K
    # K = 1 + np.max([np.sum([G.get_edge_data(x,y)['capacity'] for y in G.neighbors(x)]) for x in G.nodes])
    Gnx=G.get_nx_graph()
    K = 1 + np.max([np.sum([Gnx.get_edge_data(x,y)['weight'] for y in Gnx.neighbors(x)]) for x in Gnx.nodes])

    # Add edges for S and T
    if prior_as_index:
        for i in range(n):
            for j in range(p):
                if (i,j) in B:
                    G.add_tedge(nodeids[i,j], 0,K)
                    # G.add_edge((i,j),'T', capacity=K)
                elif (i,j) in O:
                    G.add_tedge(nodeids[i,j], K,0)
                    # G.add_edge((i,j),'S', capacity=K)
                else:
                    G.add_tedge(nodeids[i,j], ??*Rp["bkg"](img[(i,j)])[0], ??*Rp["obj"](img[(i,j)])[0])
                    # G.add_tedge(nodeids[i,j], K,K)
                    # G.add_edge((i,j),'T', capacity=??*Rp["obj"](img[(i,j)])[0])
                    # G.add_edge((i,j),'S', capacity=??*Rp["bkg"](img[(i,j)])[0])
    else:
        for i in range(n):
            for j in range(p):
                if B[i,j] == 1:
                    G.add_tedge(nodeids[i,j], 0,K)
                    # G.add_edge((i,j),'T', capacity=K)
                elif O[i,j] == 1:
                    G.add_tedge(nodeids[i,j], K,0)
                    # G.add_edge((i,j),'S', capacity=K)
                else:
                    G.add_tedge(nodeids[i,j], ??*Rp["bkg"](img[(i,j)])[0], ??*Rp["obj"](img[(i,j)])[0])
                    # G.add_edge((i,j),'T', capacity=??*Rp["obj"](img[(i,j)])[0])
                    # G.add_edge((i,j),'S', capacity=??*Rp["bkg"](img[(i,j)])[0])
    return G, Rp, nodeids

