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

#R is edge list NEdges x 2
def makeDelta1(R):
    NEdges = R.shape[0]
    NVertices = int(np.max(R))+1
    NDigits = int(np.log(NVertices)/np.log(10))
    
    #Slow 3-clique (2 simplex) finding.  TODO: Speed up
    edgeMap = {}
    for i in range(R.shape[0]):
        edgeMap[(R[i, 0], R[i, 1])] = i
    arr = np.arange(NEdges)
    I = []
    J = []
    V = []
    TriNum = 0
    for A in range(NVertices):
        for B in range(A+1, NVertices):
            for C in range(B+1, NVertices):
                e12 = (A, B)
                e13 = (A, C)
                e23 = (B, C)
                if e12 in edgeMap and e13 in edgeMap and e23 in edgeMap:
                    #The clique exists at some orientation so a row needs to be added
                    I.append([TriNum]*3)
                    J.append([edgeMap[a] for a in [e12, e13, e23]])
                    V.append([1, -1, 1])
                    TriNum += 1
    [I, J, V] = [np.array(a).flatten() for a in [I, J, V]]
    Delta1 = sparse.coo_matrix((V, (I, J)), shape=(TriNum, NEdges))
    return Delta1

#R is NEdges x 2 matrix specfiying comparisons that have been made
#W is a flat array of NEdges weights parallel to the rows of R
#Y is a flat array of NEdges specifying preferences
def doHodge(R, W, Y):
    #Step 1: Get s
    D0 = makeDelta0(R)
    wSqrt = np.sqrt(W).flatten()
    WSqrt = scipy.sparse.spdiags(wSqrt, 0, len(W), len(W))
    WSqrtRecip = scipy.sparse.spdiags(1/wSqrt, 0, len(W), len(W))
    A = WSqrt*D0
    b = WSqrt.dot(Y)
    s = lsqr(A, b)[0]
    
    #Step 2: Get local inconsistencies
    D1 = makeDelta1(R)
    B = WSqrtRecip*D1.T
    resid = Y - D0.dot(s)
    u = wSqrt*resid
    Phi = lsqr(B, u)[0]
    I = B.dot(Phi)
    
    #Step 3: Get harmonic cocycle
    H = resid - I
    
    return (s, I, H)

def doRandomFlipExperiment(N, PercentFlips):
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    I = I[np.triu_indices(N, 1)]
    J = J[np.triu_indices(N, 1)]
    NEdges = len(I)
    R = np.zeros((NEdges, 2))
    R[:, 0] = J
    R[:, 1] = I
    
#    toKeep = int(NEdges/200)
#    R = R[np.random.permutation(NEdges)[0:toKeep], :]
#    NEdges = toKeep
    
    #W = np.random.rand(NEdges)
    W = np.ones(NEdges)
    
    Y = np.ones(NEdges)
    NFlips = int(PercentFlips*len(Y))
    Y[np.random.permutation(NEdges)[0:NFlips]] = -1
    
    (s, I, H) = doHodge(R, W, Y)
    return (s, I, H)

def doRandomFlipExperimentsSame(N, PercentFlips, NTrials):
    INorm = np.zeros(NTrials)
    HNorm = np.zeros(NTrials)
    for i in range(NTrials):
        print "%i of %i"%(i, NTrials)
        (s, I, H) = doRandomFlipExperiment(N, PercentFlips)
        INorm[i] = np.sqrt(np.sum(I**2))
        HNorm[i] = np.sqrt(np.sum(H**2))
    plt.subplot(211)
    plt.plot(INorm)
    plt.title('I Norm')
    plt.subplot(212)
    plt.plot(HNorm)
    plt.title('H Norm')
    plt.show()


def doRandomFlipExperimentsVary(N, AllPercentFlips, NTrials):
    M = len(AllPercentFlips)
    INorm = np.zeros((M, NTrials))
    HNorm = np.zeros((M, NTrials))
    for i in range(M):
        print "%i of %i"%(i, M)
        for k in range(NTrials):
            (s, I, H) = doRandomFlipExperiment(N, AllPercentFlips[i])
            INorm[i, k] = np.sqrt(np.sum(I**2))
            HNorm[i, k] = np.sqrt(np.sum(H**2))
    INorm = np.mean(INorm, 1)
    HNorm = np.mean(HNorm, 1)
    plt.subplot(211)
    plt.plot(AllPercentFlips, INorm)
    plt.title('I Norm')
    plt.subplot(212)
    plt.plot(AllPercentFlips, HNorm)
    plt.title('H Norm')
    plt.show()


#Do random flip experiments
if __name__ == '__main__2':
    np.random.seed(100)
    N = 20
    doRandomFlipExperimentsVary(N, np.linspace(0, 1, 100), 50)
    
#    N = 4
#    I, J = np.meshgrid(np.arange(N), np.arange(N))
#    I = I[np.triu_indices(N, 1)]
#    J = J[np.triu_indices(N, 1)]
#    NEdges = len(I)
#    R = np.zeros((NEdges, 2))
#    R[:, 0] = J
#    R[:, 1] = I
#    
#    print makeDelta0(R).toarray()
#    print makeDelta1(R).toarray()

if __name__ == '__main__':
    R = sio.loadmat('R.mat')['R']
    [R, Y] = [R[:, 0:2], R[:, 2]]
    W = np.ones(len(Y))
    (s, I, H) = doHodge(R, W, Y)
    print np.argsort(s)
    
    normY = np.linalg.norm(Y)
    normD0s = np.linalg.norm(Y-H-I)
    normI = np.linalg.norm(I)
    normH = np.linalg.norm(H)
    print "|D0s/Y| = ", (normD0s/normY)**2
    print "Local Inconsistency = ", (normI/normY)**2
    print "Global Inconsistency = ", (normH/normY)**2
    
    #plt.plot(s)
    #plt.show()
