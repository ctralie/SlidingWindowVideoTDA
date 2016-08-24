from VideoTools import *
from TDA import *
import numpy as np
import scipy.io as sio

def runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise):
    print("PROCESSING ", filename, "....")
    persistences = []
    (XOrig, FrameDims) = loadVideo(filename)
    for i in range(NRandDraws):
        print("Random draw %i of %i"%(i, NRandDraws))
        print("Doing PCA...")
        X = getPCAVideo(XOrig + Noise*np.random.randn(XOrig.shape[0], XOrig.shape[1]))
        print("Finished PCA")
        [X, validIdx] = getTimeDerivative(X, 10)
        
        #Setup video blocks
        idxs = []
        N = X.shape[0]
        NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
        for k in range(NBlocks):
            thisidxs = np.arange(k*BlockHop, k*BlockHop+BlockLen, dtype=np.int64)
            thisidxs = thisidxs[thisidxs < N]
            idxs.append(thisidxs)
        
        for j in range(len(idxs)):
            print("Block %i of %i"%(j, len(idxs)))
            idx = idxs[j]
            Tau = win/float(dim-1)
            dT = (len(idx)-dim*Tau)/float(len(idx))
            XS = getSlidingWindowVideo(X[idx, :], dim, Tau, dT)

            #Mean-center and normalize sliding window
            XS = XS - np.mean(XS, 1)[:, None]
            XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
            
            PDs = doRipsFiltration(XS, 1)
            if len(PDs) < 2:
                continue
            if PDs[1].size > 0:
                persistences.append(np.max(PDs[1][:, 1] - PDs[1][:, 0]))
    return np.array(persistences)

def getROC(T, F, cutoffs):
    N = len(cutoffs)
    FP = np.zeros(N) #False positives
    TP = np.zeros(N) #True positives
    for i in range(N):
        FP[i] = np.sum(F > cutoffs[i])/float(F.size)
        TP[i] = np.sum(T > cutoffs[i])/float(T.size)
    return (FP, TP)

if __name__ == '__main__':
    BlockLen = 160
    BlockHop = 80
    win = 20
    dim = 20
    NRandDraws = 100
    
    for Noise in [0.1, 0.2, 0.5, 1]:
        psTrue = runExperiments("Videos/pendulum.avi", BlockLen, BlockHop, win, dim, NRandDraws, Noise)
        sio.savemat("psTrue%g.mat"%Noise, {"psTrue":psTrue})
        psFalse = runExperiments("Videos/drivingscene.mp4", BlockLen, BlockHop, win, dim, NRandDraws, Noise)
        sio.savemat("psFalse%g.mat"%Noise, {"psFalse":psFalse})
        
        (FP, TP) = getROC(psTrue, psFalse, np.linspace(0, np.sqrt(3), 1000))
        plt.clf()
        plt.plot(FP, TP, 'b')
        plt.hold(True)
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("ROC%g.svg", bbox_inches='tight')
