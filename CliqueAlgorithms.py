import numpy as np
import scipy
from scipy import sparse

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

