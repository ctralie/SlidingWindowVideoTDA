import numpy as np
import scipy
from scipy import sparse
import array

#Brute force function to check for all 3 cliques by checking
#mutual neighbors between 3 vertices
def get3CliquesBrute(Edges):
    [I, J, V] = [[], [], []]
    NVertices = len(Edges)
    MaxNum = int(NVertices*(NVertices-1)*(NVertices-2)/6)
    J = np.zeros((MaxNum, 3))
    [i, j, k] = [0, 0, 0]
    edgeNum = 0
    for i in range(NVertices):
        for j in Edges[i]:
            if j < i:
                continue
            for k in Edges[j]:
                if k < j or k < i:
                    continue
                if k in Edges[i]:
                    [a, b, c] = sorted([i, j, k])
                    J[edgeNum, :] = [Edges[a][b], Edges[a][c], Edges[b][c]]
                    edgeNum += 1
    J = J[0:edgeNum, :]
    V = np.zeros(J.shape)    
    V[:, 0] = 1
    V[:, 1] = -1
    V[:, 2] = 1
    I = np.zeros(J.shape)
    for k in range(3):
        I[:, k] = np.arange(I.shape[0])
    return (I, J, V)

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

