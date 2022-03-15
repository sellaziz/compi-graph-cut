
def _get_path(G, node1, node2):
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

def grow(G, A):
    """Expand the trees S and T. The active nodes A explore adjacent non-saturated
    edges and acquire new children from a set of free nodes. The newly acquired nodes become
    active members of the corresponding search trees. As soon as all neighbors of a given active
    node are explored the active node becomes passive. The growth stage terminates if an active
    node encounters a neighboring node that belongs to the opposite tree. In this case we detect a
    path from the source to the sink."""
    
    while A:
        active = A[0]
        for neigh in G.neighbors(active):

            if neigh not in ['S', 'T']:

                edge = G.get_edge_data(active, neigh)
                if edge['capacity'] > edge['flow']: # Saturated ?

                    if G.nodes[neigh]['tree'] == None:
                        # Add it to the growing tree (S or T)
                        G.nodes[neigh]['tree'] = G.nodes[active]['tree']
                        # Set the active node as its parent
                        G.nodes[neigh]['parent'] = active
                        # Set it as active
                        A.append(neigh)

                    elif  G.nodes[neigh]['tree'] != G.nodes[active]['tree']:
                        return _get_path(G, active, neigh)
        A.pop(0)
    return []
