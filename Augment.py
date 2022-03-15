import numpy as np

def augment(G, P, const_flow=1e-5):
    # Initialize orphans
    Orphans = []

    # Retrieve capacities, flows & residual capacities
    capacities = np.array([G.get_edge_data(P[i], P[i+1])['capacity'] for i in range(len(P)-1)])
    flows = np.array([G.get_edge_data(P[i], P[i+1])['flow'] for i in range(len(P)-1)])
    residuals = capacities - flows

    # Find the bottleneck
    df = np.min(residuals) + const_flow

    # Compute new flows & residuals
    new_flows = flows + df
    new_residuals = residuals - df
    
    # Update graph
    for i in range(len(P)-1):
        # Update flow
        G[P[i]][P[i+1]]['flow'] = new_flows[i]

        # Saturated node
        if new_residuals[i] <= 0:
            if G.nodes[P[i]]['tree'] == 'S' and G.nodes[P[i+1]]['tree'] == 'S':
                G.nodes[P[i+1]]['parent'] == None
                Orphans.append(P[i+1])
                
            if G.nodes[P[i]]['tree'] == 'T' and G.nodes[P[i+1]]['tree'] == 'T':
                G.nodes[P[i]]['parent'] == None
                Orphans.append(P[i])
    
    return Orphans
