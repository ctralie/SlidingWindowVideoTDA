from VideoTools import *
from TDA import *
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import os


def makePlot(X, PD):
    if X.size == 0:
        return
    #Self-similarity matrix
    XSqr = np.sum(X**2, 1).flatten()
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0
    D = np.sqrt(D)

    #PCA
    pca = PCA(n_components = 20)
    Y = pca.fit_transform(X)
    sio.savemat("PCA.mat", {"Y":Y})
    eigs = pca.explained_variance_

    plt.clf()
    plt.subplot(221)
    plt.imshow(D)

    plt.subplot(222)
    plotDGM(PD)
    plt.title("Max = %g"%np.max(PD[:, 1] - PD[:, 0]))

    ax = plt.subplot(223)
    ax.set_title("PCA of Sliding Window Embedding")
    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    ax.scatter(Y[:, 0], Y[:, 1], c = C)
    ax.set_aspect('equal', 'datalim')

    plt.subplot(224)
    plt.bar(np.arange(len(eigs)), eigs)
    plt.title("Eigenvalues")


def processVideo(XOrig, FrameDims, BlockLen, BlockHop, win, dim, filePrefix, doDerivative = True, doSaveVideo = True, coeff = 3):
    print("Doing PCA...")
    X = getPCAVideo(XOrig)
    print("Finished PCA")
    if doDerivative:
        [X, validIdx] = getTimeDerivative(X, 10)

    #Setup video blocks
    idxs = []
    N = X.shape[0]
    if BlockLen == -1:
        idxs = [np.arange(N)]
    else:
        NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
        print("X.shape[0] = %i, NBlocks = %i"%(X.shape[0], NBlocks))
        for k in range(NBlocks):
            thisidxs = np.arange(k*BlockHop, k*BlockHop+BlockLen, dtype=np.int64)
            thisidxs = thisidxs[thisidxs < N]
            idxs.append(thisidxs)

    PDMax = []
    XMax = np.array([])
    maxP = 0
    maxj = 0
    persistences = []
    for j in range(len(idxs)):
        print("Block %i of %i"%(j, len(idxs)))
        idx = idxs[j]
        Tau = win/float(dim-1)
        dT = (len(idx)-dim*Tau)/float(len(idx))
        XS = getSlidingWindowVideo(X[idx, :], dim, Tau, dT)

        #Mean-center and normalize sliding window
        XS = XS - np.mean(XS, 1)[:, None]
        XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
        plt.clf()
        makePlot(XMax, PDMax)
        plt.savefig("%s_Stats.png"%filePrefix)

        PDs = doRipsFiltration(XS, 1, coeff = coeff)
        if len(PDs) < 2:
            continue
        if PDs[1].size > 0:
            thisMaxP = np.max(PDs[1][:, 1] - PDs[1][:, 0])
            print "thisMaxP = ", thisMaxP
            persistences.append(thisMaxP)
            if thisMaxP > maxP:
                maxP = thisMaxP
                PDMax = PDs[1]
                XMax = XS
                maxj = j
    if len(filePrefix) > 0 and len(XMax) > 0:
        #Save an example persistence diagram for the max block
        #from the first draw
        print("Saving first block")
        plt.clf()
        makePlot(XMax, PDMax)
        plt.savefig("%s_Stats.png"%filePrefix)
        if doSaveVideo:
            saveVideo(XOrig[idxs[maxj], :], FrameDims, "%s_max.ogg"%filePrefix)
    return (PDMax, XMax, maxP, maxj, persistences)

def runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent):
    print("PROCESSING ", filename, "....")
    persistences = []
    (XOrig, FrameDims) = loadVideo(filename)
    for i in range(NRandDraws):
        filePrefix = "%s_%g_%i"%(filename, Noise, BlurExtent)
        print("filePrefix = %s"%filePrefix)
        print("Random draw %i of %i"%(i, NRandDraws))
        if BlurExtent > 0:
            XSample = simulateCameraShake(XOrig, FrameDims, BlurExtent)
        else:
            XSample = XOrig
        XSample = XSample + Noise*np.random.randn(XOrig.shape[0], XOrig.shape[1])
        (PDMax, XMax, maxP, maxj, p) = processVideo(XSample, FrameDims, BlockLen, BlockHop, win, dim, filePrefix, doSaveVideo = True)
        print("maxP = %g"%maxP)
        persistences = persistences + p
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
    dim = 40
    NRandDraws = 200

    files = {'pullups':'Videos/pullups.avi', 'explosions':'Videos/explosions.mp4', 'heartbeat':'Videos/heartcrop.avi'}
    #files = {'pendulum':'Videos/pendulum.avi'}
    #files = {'driving':"Videos/drivingscene.mp4"}
    #files = {'explosions':'Videos/explosions.mp4'}
    for name in files:
        filename = files[name]
        for Noise in [0]:#[0.001, 0.1, 0.5, 1]:
            for BlurExtent in [0, 20, 40, 80]:#[2, 10, 20, 40]:
                if os.path.exists("psTrue%s_%g_%i.mat"%(name, Noise, BlurExtent)):
                    print("Loading precomputed psTrue %s %g %i"%(name, Noise, BlurExtent))
                    psTrue = sio.loadmat("psTrue%s_%g_%i.mat"%(name, Noise, BlurExtent))['psTrue']
                else:
                    psTrue = runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent)
                    sio.savemat("psTrue%s_%g_%i.mat"%(name, Noise, BlurExtent), {"psTrue":psTrue})

                if os.path.exists("psFalse_%g_%i.mat"%(Noise, BlurExtent)):
                    psFalse = sio.loadmat("psFalse_%g_%i.mat"%(Noise, BlurExtent))['psFalse']
                else:
                    psFalse = runExperiments("Videos/drivingscene.ogg", BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent)
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
