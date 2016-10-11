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
    
#Brute force function to check for all 3 cliques by checking
#mutual neighbors between 3 vertices
def get3CliquesBrute(Edges):
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

#Recursive function to find all of the maximal cliques
def BronKerbosch(C, U, X, E, Cliques, callOrder = 0, verbose = False):
    if len(U) == 0 and len(X) == 0:
        if verbose:
            print "%sFound clique "%("\t"*callOrder), C
        C = sorted(C)
        Cliques.append(C)
        return
    #Choose a pivot vertex u in U union X
    i = np.random.randint(len(U)+len(X))
    upivot = 0
    if i >= len(U):
        upivot = X[i-len(U)]
    else:
        upivot = U[i]
    UList = []
    #For each vertex v in U \ N(u)
    for u in U:
        if not (E[u, upivot] or E[upivot, u]):
            UList.append(u)
    for v in UList:
        #UNew = U intersect N(v)
        UNew = []
        for u in U:
            if E[u, v] or E[v, u]:
                UNew.append(u)
        #XNew = X intersect N(v)
        XNew = []
        for x in X:
            if E[x, v] or E[v, x]:
                XNew.append(x)
        if verbose:
            print "%sBK(%s, %s, %s)"%("\t"*callOrder, C + [v], UNew, XNew)
        BronKerbosch(C + [v], UNew, XNew, E, Cliques, callOrder + 1, verbose)
        U.remove(v)
        X.append(v)

#Extract 3 cliques from maximal cliques, given an array 
#with edge indices
def get3CliquesFromMaxCliques(Cliques, E):
    cliques = set()
    [I, J, V] = [[], [], []]
    for c in Cliques:
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                for k in range(j+1, len(c)):
                    cliques.add((c[i], c[j], c[k]))
    for c in cliques:
        I.append(3*[len(I)])
        J.append([E[c[0], c[1]]-1, E[c[0], c[2]]-1, E[c[1], c[2]]-1])
        V.append([1.0, -1.0, 1.0])
    [I, J, V] = [np.array(a).flatten() for a in [I, J, V]]
    return (I, J, V)

#Compare boundary matrices up to a permutation of the rows
def compareBoundaryMatricesModPerm(A, B):
    setA = set()
    for i in range(A.shape[0]): #Assuming csr
        a = A[i, :]
        a = a.tocoo()
        arr = tuple(a.col.tolist() + a.data.tolist())
        setA.add(arr)
    setB = set()
    for i in range(B.shape[0]):
        b = B[i, :]
        b = b.tocoo()
        arr = tuple(b.col.tolist() + b.data.tolist())
        setB.add(arr)
    if len(setA.difference(setB)) > 0 or len(setB.difference(setA)) > 0:
        return False
    return True

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
    
    (I, J, V) = get3CliquesBrute(Edges)
    TriNum = len(I)/3
    Delta1 = sparse.coo_matrix((V, (I, J)), shape = (TriNum, NEdges)).tocsr()
    
    [C, X, Cliques] = [[], [], []]
    E = sparse.coo_matrix((1+np.arange(NEdges), (R[:, 0], R[:, 1])), shape = (NVertices, NVertices)).tocsr()
    
    U = range(NVertices)
    #print "BK(%s, %s, %s)"%(C, U, X)
    BronKerbosch(C, U, X, E, Cliques, verbose = False)
    TriNum = len(I)/3
    [I, J, V] = get3CliquesFromMaxCliques(Cliques, E)
    Delta1B = sparse.coo_matrix((V, (I, J)), shape = (TriNum, NEdges)).tocsr()
    
    print "Matrices are the same: ", compareBoundaryMatricesModPerm(Delta1, Delta1B)
    print np.sum(Delta1.toarray() - Delta1B.toarray())
    
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
    np.random.seed(10)
    N = 100
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I    
    R = R[np.random.permutation(R.shape[0])[0:R.shape[0]/2], :]
    makeDelta1(R)
    #print makeDelta0(R).toarray()
    #print makeDelta1(R).toarray()
