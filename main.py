import numpy as np
import networkx as nx


def image2graph(img, O, B, nbins=10):
    """Convert the iput image into a graph for the segmentation part.
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
    A = O + B

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
        print(A)
        for neigh in G.neighbors(active):
            if neigh not in ['S', 'T']:
                edge = G.get_edge_data(active, neigh)

                if edge['capacity'] < edge['flow']:
                    pass # Saturated egde

                if G.nodes[neigh]['tree'] == None:
                    # Add it to the growing tree (S or T)
                    G.nodes[neigh]['tree'] = G.nodes[active]['tree']
                    # Set the active node as its parent
                    G.nodes[neigh]['parent'] = active
                    # Set it as active
                    A.append(neigh)

                elif  G.nodes[neigh]['tree'] != G.nodes[active]['tree']:
                    return get_path(G, active, neigh)
        A.pop(0)
    return []

def augment(S,T):
    pass

def adopt(O):
    pass

def segment():
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