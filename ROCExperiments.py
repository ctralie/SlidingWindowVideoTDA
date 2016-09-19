from VideoTools import *
from TDA import *
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import os


def makePlot(X, PD):
    #Self-similarity matrix
    XSqr = np.sum(X**2, 1).flatten()
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0
    D = np.sqrt(D)

    #PCA
    pca = PCA(n_components = 20)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_

    plt.clf()
    plt.subplot(221)
    plt.imshow(D)

    plt.subplot(222)
    plotDGM(PD)
    plt.title("Max = %g"%np.max(PD[:, 1] - PD[:, 0]))

    ax = plt.subplot(223)
    ax.set_title("PCA of Sliding Window Embedding")
    ax.scatter(Y[:, 0], Y[:, 1])
    ax.set_aspect('equal', 'datalim')
    
    plt.subplot(224)
    plt.bar(np.arange(len(eigs)), eigs)
    plt.title("Eigenvalues")

def runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent):
    print("PROCESSING ", filename, "....")
    persistences = []
    (XOrig, FrameDims) = loadVideo(filename)
    for i in range(NRandDraws):
        print("Random draw %i of %i"%(i, NRandDraws))
        print("Doing PCA...")
        XSample = simulateCameraShake(XOrig, FrameDims, BlurExtent)
        XSample = XSample + Noise*np.random.randn(XOrig.shape[0], XOrig.shape[1])
        X = getPCAVideo(XSample)
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
        
        PDMax = []
        XMax = []
        maxP = 0
        maxj = 0
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
                thisMaxP = np.max(PDs[1][:, 1] - PDs[1][:, 0])
                persistences.append(thisMaxP)
                if thisMaxP > maxP:
                    maxP = thisMaxP
                    PDMax = PDs[1]
                    XMax = XS
                    maxj = j
        if i == 0:
            #Save an example persistence diagram for the max block
            #from the first draw
            print("Saving first block")
            plt.clf()
            makePlot(XMax, PDMax)
            plt.savefig("%s_%g_%i_Stats.png"%(filename, Noise, BlurExtent))
            saveVideo(XSample[idxs[maxj], :], FrameDims, "%s_%g_%i_max.ogg"%(filename, Noise, BlurExtent))
    return np.array(persistences)

def getROC(T, F):
    values = np.sort(np.array(T.flatten().tolist() + F.flatten().tolist()))
    N = len(values)
    FP = np.zeros(N) #False positives
    TP = np.zeros(N) #True positives
    for i in range(N):
        FP[i] = np.sum(F >= values[i])/float(F.size)
        TP[i] = np.sum(T >= values[i])/float(T.size)
    return (FP, TP)

if __name__ == '__main__':
    BlockLen = 160
    BlockHop = 80
    win = 20
    dim = 20
    NRandDraws = 50
    
    #files = {'pendulum':'Videos/pendulum.avi', 'heart':'Videos/heartvariations.mp4', 'butterflies':'Videos/butterflies.mp4'}
    files = {'pendulum':'Videos/pendulum.avi'}
    #files = {'driving':"Videos/drivingscene.mp4"}
    #files = {'explosions':'Videos/explosions.mp4'}
    for name in files:
        filename = files[name]
        for Noise in [0.001, 0.1, 0.5, 1]:
            for BlurExtent in [2, 10, 20, 40]:
                psTrue = runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent)
                sio.savemat("psTrue%s_%g_%i.mat"%(name, Noise, BlurExtent), {"psTrue":psTrue})
                
                if os.path.exists("psFalse_%g_%i.mat"%(Noise, BlurExtent)):
                    psFalse = sio.loadmat("psFalse_%g_%i.mat"%(Noise, BlurExtent))['psFalse']
                else:
                    psFalse = runExperiments("Videos/drivingscene.mp4", BlockLen, BlockHop, win, dim, 50, Noise, BlurExtent)
                    sio.savemat("psFalse_%g_%i.mat"%(Noise, BlurExtent), {"psFalse":psFalse})
                
                #Plot ROC curve
                (FP, TP) = getROC(psTrue, psFalse)
                plt.clf()
                plt.plot(FP, TP, 'b')
                plt.hold(True)
                plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.savefig("ROC_%s_%g_%i.svg"%(name, Noise, BlurExtent), bbox_inches='tight')
                
                #Plot histogram
                plt.clf()
                plt.hold(True)
                plt.hist(psTrue.flatten(), 10, facecolor='blue')
                plt.hist(psFalse.flatten(), 10, facecolor='red')
                plt.savefig("Hists_%s_%g_%i.svg"%(name, Noise, BlurExtent), bbox_inches='tight')
