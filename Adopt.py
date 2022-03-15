
def _test_origin(G, node):
    """Return the origin of the input node, e.g its furthest 
    parent connected through non-saturated edges."""
    origin = node
    while True:
        if origin in ['S','T']:
            return True
        
        # Has a parent
        parent = G.nodes[origin]['parent']
        if not parent:
            return False
        
        # Edge to parent not saturated
        edge = G.get_edge_data(origin,parent)
        if edge['capacity'] > edge['flow']:
            origin = G.nodes[origin]['parent']
        else:
            return False

def adopt(G, Orphans, A):
    """
    Adoption stage: Reconstruct search trees by adopting orphans.
    During the augmentation stage, some edges became saturated. 
    As a consequence, the source and target search trees broke 
    down to forests, with orphans as roots of some of its trees. 
    The goal of the adoption stage is to restore single-tree 
    structure of sets S and T.
    """
    while Orphans:
        p = Orphans.pop(0)
        
        # Try to find a new valid parent
        parent_found = False
        for q in G.neighbors(p):
            if G.nodes[p]['tree'] == G.nodes[q]['tree']:
                edge = G.get_edge_data(q,p)
                if edge['capacity'] > edge['flow']:
                    if _test_origin(G, q):
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
            
            G.nodes[p]['tree'] = None
            if p in A:
                A.remove(p)

        Orphans = list(set(Orphans))

    return A
