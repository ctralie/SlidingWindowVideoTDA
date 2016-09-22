import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh

#R is a NEdges x 2 matrix specifying edges, where orientation
#is taken from the first column to the second column
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

#R is NEdges x 2 matrix specfiying comparisons that have been made
#W is a flat array of NEdges weights parallel to the rows of R
#Y is a flat array of NEdges specifying preferences
def doHodge(R, W, Y):
    D0 = makeDelta0(R)
    WSqrt = np.sqrt(W)
    WSqrt = scipy.sparse.spdiags(WSqrt, 0, len(W), len(W))
    A = WSqrt*D0
    b = WSqrt.dot(Y)
    s = lsqr(A, b)[0]
    return s
    
if __name__ == '__main__':
    N = 10
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N)]
    J = J[np.triu_indices(N)]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = I
    R[:, 1] = J
    W = np.random.rand(NEdges)#np.ones(NEdges)
    Y = np.ones(NEdges)
    s = doHodge(R, W, Y)
    
    plt.plot(s, '.')
    plt.show()
