import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
import time

#R is a NEdges x 2 matrix specifying edges, where orientation
#is taken from the first column to the second column
#R specifies the "natural orientation" of the edges, with the
#understanding that the ranking will be specified later
#It is assumed that there is at least one edge incident
#on every vertex
def makeDelta0(R):
    NVertices = np.max(R) + 1
    NEdges = R.shape[0]
    
    #Two entries per edge
    I = np.zeros((NEdges, 2))
    I[:, 0] = np.arange(NEdges)
    I[:, 1] = np.arange(NEdges)
    I = I.flatten()
    
    J = R[:, 0:2].flatten()
    
    V = np.zeros((NEdges, 2))
    V[:, 0] = -1
    V[:, 1] = 1
    V = V.flatten()
    
    Delta = sparse.coo_matrix((V, (I, J)), shape=(NEdges, NVertices)).tocsr()
    return Delta
    
def get3Cliques(Edges):
    [I, J, V] = [[], [], []]
    NVertices = len(Edges)
    for i in range(NVertices):
        for j in Edges[i]:
            if j < i:
                continue
            for k in Edges[j]:
                if k < j or k < i:
                    continue
                if k in Edges[i]:
                    TriNum = len(I)
                    I.append([TriNum]*3)
                    [a, b, c] = sorted([i, j, k])
                    J.append([Edges[a][b], Edges[a][c], Edges[b][c]])
                    V.append([1, -1, 1])
    [I, J, V] = [1.0*np.array(a).flatten() for a in [I, J, V]]
    return (I, J, V)
        

#R is edge list NEdges x 2
#It is assumed that there is at least one edge incident
#on every vertex
def makeDelta1(R):
    NEdges = R.shape[0]
    NVertices = int(np.max(R))+1
    #Make a list of edges for fast lookup
    Edges = []
    for i in range(NVertices):
        Edges.append({})
    for i in range(R.shape[0]):
        [a, b] = [int(R[i, 0]), int(R[i, 1])]
        Edges[a][b] = i
        Edges[b][a] = i    
    
    (I, J, V) = get3Cliques(Edges)
    TriNum = len(I)/3
    Delta1 = sparse.coo_matrix((V, (I, J)), shape = (TriNum, NEdges))    
    return Delta1

#R is NEdges x 2 matrix specfiying comparisons that have been made
#W is a flat array of NEdges weights parallel to the rows of R
#Y is a flat array of NEdges specifying preferences
def doHodge(R, W, Y, verbose = False):
    #Step 1: Get s
    if verbose:
        print "Making Delta0..."
    tic = time.time()
    D0 = makeDelta0(R)
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc-tic, " seconds"
    wSqrt = np.sqrt(W).flatten()
    WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
    WSqrtRecip = scipy.sparse.spdiags(1/wSqrt, 0, len(W), len(W))
    A = WSqrt*D0
    b = WSqrt.dot(Y)
    s = lsqr(A, b)[0]
    
    #Step 2: Get local inconsistencies
    if verbose:
        print "Making Delta1..."
    tic = time.time()
    D1 = makeDelta1(R)
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc-tic, " seconds"
    B = WSqrtRecip*D1.T
    resid = Y - D0.dot(s)  #This has been verified to be orthogonal under <resid, D0*s>_W
    
    u = wSqrt*resid
    if verbose:
        print "Solving for Phi..."
    tic = time.time()
    Phi = lsqr(B, u)[0]
    toc = time.time()
    if verbose:
        print "Elapsed Time: ", toc - tic, " seconds"
    I = WSqrtRecip.dot(B.dot(Phi)) #Delta1* dot Phi, since Delta1* = (1/W) Delta1^T
    
    #Step 3: Get harmonic cocycle
    H = resid - I
    return (s, I, H)

def getWNorm(X, W):
    return np.sqrt(np.sum(W*X*X))


#Do an experiment with a full 4-clique to make sure 
#that delta0 and delta1 look right
if __name__ == '__main__':
    np.random.seed(50)
    N = 4
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I    
    print makeDelta0(R).toarray()
    print makeDelta1(R).toarray()
