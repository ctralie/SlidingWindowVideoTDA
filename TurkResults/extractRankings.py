"""Convert the numbers users inputted into their true indices"""
import numpy as np
import scipy.io as sio

if __name__ == "__main__":
    R1 = sio.loadmat("Subject2Rankings1.mat")["R"]
    np.random.seed(100)
    IDs = np.random.permutation(999)
    R = np.array(R1, dtype = np.float32)
    for i in range(R1.shape[0]):
        id1 = IDs[R1[i, 0]]
        id2 = IDs[R1[i, 1]]
        if R1[i, 2] == id1:
            R[i, 2] = 1
        elif R1[i, 2] == id2:
            R[i, 2] = -1
        else:
            print "Nonmatching number %i (%i or %i)"%(R1, id1, id2)
    print R
    sio.savemat("Subject2Rankings1.mat", {"R":R})
