import networkx

def image2graph(img):
    return graph

def growth_stage(A,S,T):
    global P
    return 0 # path

def augment(S,T):
    global P
    pass

def adopt(O):
    pass

def segment():
    global P
    # initialize: S = {s}, T = {t}, A = {s, t}, O = ∅
    while True:
        # grow S or T to find an augmenting path P from s to t
        P_isempty=growth_stage()
        if P_isempty:
            break
        # if P = ∅ terminate
        # augment on P
        augment()
        # adopt orphans
        adopt()
    # end while
    pass