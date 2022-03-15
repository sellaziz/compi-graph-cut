import numpy as np
import networkx as nx
import skimage
import matplotlib.pyplot as plt

def initialize_priors(img_painted):
    """Image painted with red for Object and blue for Source.
        Return the Object and Background pixels."""
    red, green, blue = np.transpose(img_painted, (2,0,1))
    O_mask = (red > 200) * (green < 100) * (blue < 100)
    B_mask = (red < 80) * (green < 80) * (blue > 140)
    O = [tuple(idx) for idx in np.argwhere(O_mask)]
    B = [tuple(idx) for idx in np.argwhere(B_mask)]
    return O, B

def image2graph(img, O, B, nbins=10):
    """Convert the input image into a graph for the segmentation part.
        O, B: Object and Background pixels as list of tuple.
    """

    def get_prob(hist, value):
        """Get probability of the input value from the input histogram."""
        counts, bin_edges = hist
        bin = np.argmax(bin_edges - value > 0) - 1
        if bin == -1:
            return 0
        else:
            return counts[bin] / counts.sum()

    def Rp(p, label, eps=1e-10):
        """The regional term for a pixel p, with eps to prevent infinity."""
        proba = get_prob(hists[label], img[p])
        return -np.log(proba or eps)

    def dist(p, q):
        """Distance L1 between pixels p and q."""
        return abs(p[0]-q[0]) + abs(p[1]-q[1])

    def Bpq(p, q, sig=1):
        """The boundary properties term between pixels p and q."""
        return np.exp(-(img[p]-img[q])**2/(2*sig**2)) / dist(p,q)

    def get_capacity(p, q, λ=1, K=None):
        """Compute weight between nodes p and q.
        When q is the source S or the sink T, K must be given.
        """
        if q == 'S':
            if p in O:
                return K
            if p in B:
                return 0
            return λ*Rp(p,"bkg")
        
        if q == 'T':
            if p in O:
                return 0
            if p in B:
                return K
            return λ*Rp(p,"obj")
        
        return Bpq(p,q)
    
    n,p = img.shape
    hist_O = np.histogram(O, bins=nbins, density=True)
    hist_B = np.histogram(B, bins=nbins, density=True)
    hists = dict(obj=hist_O, bkg=hist_B)

    # Initialize graph without S and T nodes
    G = nx.Graph()
    for i in range(n):
        for j in range(p):
            if i+1 < n:
                G.add_edge((i,j),(i+1,j), capacity=get_capacity((i,j),(i+1,j)))
            if j+1 < p:
                G.add_edge((i,j),(i,j+1), capacity=get_capacity((i,j),(i,j+1)))
    
    # Compute K
    K = 0
    for x in G.nodes:
        K = max(K, np.sum([G.get_edge_data(x,y)['capacity'] for y in G.neighbors(x)]))
    K += 1

    # Add edges for S and T
    for i in range(n):
        for j in range(p):
            G.add_edge((i,j),'S', capacity=get_capacity((i,j),'S',K=K))
            G.add_edge((i,j),'T', capacity=get_capacity((i,j),'T',K=K))
    
    # Add other attributes
    nx.set_node_attributes(G, None, "parent")
    nx.set_node_attributes(G, None, "tree")
    nx.set_edge_attributes(G, 0, "flow")

    # Initialize O and B attributes
    for node in O:
        G.nodes[node]['parent'] = 'S'
        G.nodes[node]['tree'] = 'S'
    for node in B:
        G.nodes[node]['parent'] = 'T'
        G.nodes[node]['tree'] = 'T'

    # Initial active nodes
    A = ['S','T']

    return G, A

def growth_stage(G, A):
    """Expand the trees S and T. The active nodes A explore adjacent non-saturated
    edges and acquire new children from a set of free nodes. The newly acquired nodes become
    active members of the corresponding search trees. As soon as all neighbors of a given active
    node are explored the active node becomes passive. The growth stage terminates if an active
    node encounters a neighboring node that belongs to the opposite tree. In this case we detect a
    path from the source to the sink."""

    def get_path(node1, node2):
        """Construct the path from the graph G and the 
        touching nodes from trees S and T."""
        # Recover parents recursively
        parents1, parents2 = [node1], [node2]
        while G.nodes[parents1[-1]]['parent']:
            parents1.append(G.nodes[parents1[-1]]['parent'])
        while G.nodes[parents2[-1]]['parent']:
            parents2.append(G.nodes[parents2[-1]]['parent'])

        # Create the path from the source S to the sink T
        if G.nodes[node1]['tree'] == 'S':
            return parents1[::-1] + parents2
        else:
            return parents2[::-1] + parents1
    
    while A:
        active = A[0]
        # print(A)
        for neigh in G.neighbors(active):
            if neigh not in ['S', 'T']:
                edge = G.get_edge_data(active, neigh)

                if edge['capacity'] <= edge['flow']:
                    pass # Saturated egde

                if G.nodes[neigh]['tree'] == None:
                    # Add it to the growing tree (S or T)
                    G.nodes[neigh]['tree'] = G.nodes[active]['tree']
                    # Set the active node as its parent
                    G.nodes[neigh]['parent'] = active
                    # Set it as active
                    A.append(neigh)

                elif  G.nodes[neigh]['tree'] != G.nodes[active]['tree']:
                    return get_path(active, neigh)
        A.pop(0)
    return []

def augment(G, P):
    # Initialize orphans
    Orphans = []

    # Retrieve capacities, flows & residual capacities
    capacities = np.array([G.get_edge_data(P[i], P[i+1])['capacity'] for i in range(len(P)-1)])
    flows = np.array([G.get_edge_data(P[i], P[i+1])['flow'] for i in range(len(P)-1)])
    residuals = capacities - flows

    # Find the bottleneck
    df = np.min(residuals)

    # Compute new flows & residuals
    new_flows = flows + df
    new_residuals = residuals - df
    
    # Update graph
    for i in range(len(P)-1):
        # Update flow
        G[P[i]][P[i+1]]['flow'] = new_flows[i]

        # Saturated node
        if new_residuals[i] == 0:
            if G.nodes[P[i]]['tree'] == 'S' and G.nodes[P[i+1]]['tree'] == 'S':
                G.nodes[P[i+1]]['parent'] == None
                Orphans.append(P[i+1])
                
            if G.nodes[P[i]]['tree'] == 'T' and G.nodes[P[i+1]]['tree'] == 'T':
                G.nodes[P[i]]['parent'] == None
                Orphans.append(P[i])
    
    return Orphans


def test_origin(G, O, node):
    """Return the origin of the input node, e.g its furthest 
    parent connected through non-saturated edges."""
    origin = node
    while True:
        if G.nodes[origin]['parent'] in O:
            return False
        else:
            origin = G.nodes[origin]['parent']
            if origin in ['S','T']:
                return True

def adopt(Orphans, G, A):
  """
  Adoption stage: Reconstruct search trees by adopting orphans.
  During the augmentation stage, some edges became saturated. 
  As a consequence, the source and target search trees broke 
  down to forests, with orphans as roots of some of its trees. 
  The goal of the adoption stage is to restore single-tree 
  structure of sets S and T.
  """
  while Orphans:
    p = Orphans.pop()
    
    # Try to find a new valid parent
    parent_found = False
    for q in G.neighbors(p):
        if G.nodes[p]['tree'] == G.nodes[q]['tree']:
            edge = G.get_edge_data(q,p)
            if edge['capacity'] > edge['flow']:
                if test_origin(G, Orphans, q):
                    G.nodes[p]['parent'] = q
                    parent_found = True
                    break
    
    # If no parent has been found
    if not parent_found:
        for q in G.neighbors(p):
            if G.nodes[p]['tree'] == G.nodes[q]['tree']:
                edge = G.get_edge_data(q,p)
                if edge['capacity'] > edge['flow']:
                    if q not in A:
                        A.append(q)
            if G.nodes[q]['parent'] == p:
                Orphans.append(q)
                G.nodes[q]['parent'] = None
        
        G.nodes[p]['tree'] == None
        if p in A:
            A.remove(p)

  return G


def segment(G):
    # initialize: S = {s}, T = {t}, A = {s, t}, O = ∅
    while True:
        # grow S or T to find an augmenting path P from s to t
        P = growth_stage()
        if not P:
            break
        # if P = ∅ terminate
        # augment on P
        augment()
        # adopt orphans
        adopt()
    # end while
    pass

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
    P=growth_stage(G, A)
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
