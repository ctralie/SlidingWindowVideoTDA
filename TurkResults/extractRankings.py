"""Convert the numbers users inputted into their true indices"""
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse

if __name__ == "__main__":
    R1 = sio.loadmat("MixRankingsOrig.mat")["R"]
    np.random.seed(100)
    IDs = np.random.permutation(999)
    I = []
    J = []
    V = []
    N = 0
    for i in range(R1.shape[0]):
        id1 = IDs[R1[i, 0]]
        id2 = IDs[R1[i, 1]]
        N = max(N, max(R1[i, 0], R1[i, 1]))
        if R1[i, 2] == id1:
            I.append(R1[i, 0])
            J.append(R1[i, 1])
            V.append(1.0)
        elif R1[i, 2] == id2:
            I.append(R1[i, 0])
            J.append(R1[i, 1])
            V.append(-1.0)
        else:
            print("Nonmatching number i = %i, %i (%i or %i)"%(i, R1[i, 2], id1, id2))
    N += 1
    R = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    R = R.toarray()
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    R = R[I > J]
    I1, J1 = np.meshgrid(np.arange(N), np.arange(N))
    I1 = I1[I > J]
    J1 = J1[I > J]
    RRet = np.zeros((I1.size, 3))
    RRet[:, 0] = I1
    RRet[:, 1] = J1
    RRet[:, 2] = -R  #TODO: I should figure out this negative sign
    sio.savemat("MixRankings.mat", {"R":RRet})
