from VideoTools import *
from TDA import *
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import os


def getPeriodicityScores(I1Z2, I1Z3, I2):
    (Z2H1Max1, Z2H1Max2, Z3H1Max1, Z3H1Max2, H2Max1) = (0, 0, 0, 0, 0)
    idx = np.argsort(I1Z3[:, 0] - I1Z3[:, 1])
    if len(idx) > 0:
        Z3H1Max1 = I1Z3[idx[0], 1] - I1Z3[idx[0], 0]
    if len(idx) > 1:
        Z3H1Max2 = I1Z3[idx[1], 1] - I1Z3[idx[1], 0]
    idx = np.argsort(I1Z2[:, 0] - I1Z2[:, 1])
    if len(idx) > 0:
        Z2H1Max1 = I1Z2[idx[0], 1] - I1Z2[idx[0], 0]
    if len(idx) > 1:
        Z2H1Max2 = I1Z2[idx[1], 1] - I1Z2[idx[0], 0]
    idx = np.argsort(I2[:, 0] - I2[:, 1])
    if len(idx) > 0:
        H2Max1 = I2[idx[0], 1] - I2[idx[0], 0]
    #Periodicity Score
    PScore = (Z3H1Max1 - Z3H1Max2)/np.sqrt(3)
    PScore = max((Z2H1Max1 - Z2H1Max2)/np.sqrt(3), PScore)
    #Harmonic Subscore
    HSubscore = 0
    if Z3H1Max1 > 0:
        1 - Z2H1Max1/Z3H1Max1
    elif Z2H1Max1 > 0:
        HSubscore = 1
    #Quasiperiodicity Score
    QPScore = np.sqrt(Z3H1Max2*H2Max1/3.0)
    return (PScore, HSubscore, QPScore)

def makePlot(X, I1Z2, I1Z3, I2):
    if X.size == 0:
        return
    #Self-similarity matrix
    XSqr = np.sum(X**2, 1).flatten()
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    (PScore, HSubscore, QPScore) = getPeriodicityScores(I1Z2, I1Z3, I2)

    #PCA
    pca = PCA(n_components = 20)
    Y = pca.fit_transform(X)
    sio.savemat("PCA.mat", {"Y":Y})
    eigs = pca.explained_variance_

    plt.clf()
    plt.subplot(221)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.title('SSM')

    plt.subplot(222)
    H1 = plotDGM(I1Z3, color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(I2, color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.title("PScore = %g, QPScore = %g"%(PScore, QPScore))

    ax = plt.subplot(223)
    ax.set_title("PCA of Sliding Window Embedding")
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    ax.scatter(Y[:, 0], Y[:, 1], c = C, edgecolor='none')
    ax.set_axis_bgcolor((0.15, 0.15, 0.15))
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_aspect('equal', 'datalim')

    plt.subplot(224)
    plt.bar(np.arange(len(eigs)), eigs)
    plt.title("Eigenvalues")


def processVideo(XOrig, FrameDims, BlockLen, BlockHop, win, dim, filePrefix, doDerivative = True, doSaveVideo = True):
    print("Doing PCA...")
    X = getPCAVideo(XOrig)
    print("Finished PCA")
    if doDerivative:
        [X, validIdx] = getTimeDerivative(X, 10)
    #Setup video blocks
    idxs = []
    N = X.shape[0]
    if BlockLen > N:
        BlockLen = -1
    if BlockLen == -1:
        idxs = [np.arange(N)]
    else:
        NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
        print("X.shape[0] = %i, NBlocks = %i"%(X.shape[0], NBlocks))
        for k in range(NBlocks):
            thisidxs = np.arange(k*BlockHop, k*BlockHop+BlockLen, dtype=np.int64)
            thisidxs = thisidxs[thisidxs < N]
            idxs.append(thisidxs)

    PScores = []
    QPScores = []
    for j in range(len(idxs)):
        print("Block %i of %i"%(j, len(idxs)))
        idx = idxs[j]
        Tau = win/float(dim-1)
        dT = (len(idx)-dim*Tau)/float(len(idx))
        XS = getSlidingWindowVideo(X[idx, :], dim, Tau, dT)

        #Mean-center and normalize sliding window
        XS = XS - np.mean(XS, 1)[:, None]
        XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]

        [I1Z2, I1Z3, I2] = [np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])]
        PDs2 = doRipsFiltration(XS, 2, coeff=2)
        PDs3 = doRipsFiltration(XS, 1, coeff=3)
        if len(PDs2) > 1:
            I1Z2 = PDs2[1]
            if I1Z2.size == 0:
                I1Z2 = np.array([[0, 0]])
        if len(PDs3) > 1:
            I1Z3 = PDs3[1]
            if I1Z3.size == 0:
                I1Z3 = np.array([[0, 0]])
        if len(PDs2) > 2:
            I2 = PDs2[2]
            if I2.size == 0:
                I2 = np.array([[0, 0]])
        (PScore, HSubscore, QPScore) = getPeriodicityScores(I1Z2, I1Z3, I2)
        PScores += [PScore]
        QPScores += [QPScore]

        if j == 0 and len(filePrefix) > 0:
            #Save an example persistence diagram for a block in the first draw
            plt.clf()
            makePlot(XS, I1Z2, I1Z3, I2)
            plt.savefig("%s_Stats.svg"%filePrefix, bbox_inches='tight')
            if doSaveVideo:
                saveVideo(XOrig[idxs[0], :], FrameDims, "%s.ogg"%filePrefix)
    return (PScores, QPScores)

def runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent, ByteErrorFrac):
    print("PROCESSING ", filename, "....")
    PScores = []
    QPScores = []
    (XOrig, FrameDims) = loadVideo(filename)
    N = XOrig.shape[0]
    BlockLen = min(BlockLen, N)
    NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
    ActualRandDraws = int(np.ceil(float(NRandDraws)/NBlocks))
    print "ActualRandDraws = ", ActualRandDraws
    for i in range(ActualRandDraws):
        doSaveVideo = False
        if i == 0:
            doSaveVideo = True
        XSample = np.array(XOrig)
        ThisFrameDims = FrameDims
        filePrefix = ""
        if i == 0:
            filePrefix = "%s_%g_%i_%g"%(filename, Noise, BlurExtent, ByteErrorFrac)
        print("Random draw %i of %i"%(i, ActualRandDraws))
        if ByteErrorFrac > 0:
            (XSample, ThisFrameDims) = simulateByteErrors(XSample, ThisFrameDims, ByteErrorFrac)
        if BlurExtent > 0:
            XSample = simulateCameraShake(XSample, ThisFrameDims, BlurExtent)
        XSample = XSample + Noise*np.random.randn(XSample.shape[0], XSample.shape[1])
        (p, qp) = processVideo(XSample, ThisFrameDims, BlockLen, BlockHop, win, dim, filePrefix, doSaveVideo)
        PScores += p
        QPScores += qp
    return (np.array(PScores), np.array(QPScores))

def getROC(T, F):
    values = np.sort(np.array(T.flatten().tolist() + F.flatten().tolist()))
    N = len(values)
    FP = np.zeros(N) #False positives
    TP = np.zeros(N) #True positives
    for i in range(N):
        FP[i] = np.sum(F >= values[i])/float(F.size)
        TP[i] = np.sum(T >= values[i])/float(T.size)
    return (FP, TP)

def plotROC(psTrue, psFalse, filename):
    #Plot ROC curve
    (FP, TP) = getROC(psTrue, psFalse)

    plt.subplot(121)
    plt.plot(FP, TP, 'b')
    plt.hold(True)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    #Plot histogram
    plt.subplot(122)
    plt.hold(True)
    plt.hist(psTrue.flatten(), 10, facecolor='blue')
    plt.hist(psFalse.flatten(), 10, facecolor='red')
    plt.savefig("Hists_ROC%s.svg"%filename, bbox_inches='tight')

if __name__ == '__main__':
    BlockLen = 300
    BlockHop = 10
    win = 30
    dim = 40
    NRandDraws = 100

    files = {'quasiperiodic':'Videos/QuasiperiodicCircles.ogg', 'pendulum':'Videos/pendulum.avi', 'explosions':'Videos/explosions.mp4', 'heartbeat':'Videos/heartcrop.avi', 'driving':"Videos/drivingscene.mp4", 'explosions':'Videos/explosions.mp4'}

    params = [{'Noise':0, 'BlurExtent':0, 'ByteError':0}]
    for Noise in [0.5, 1, 2]:
        params.append({'Noise':Noise, 'BlurExtent':0, 'ByteError':0})
    for BlurExtent in [20, 40, 80]:
        params.append({'Noise':0, 'BlurExtent':BlurExtent, 'ByteError':0})
    for ByteError in [0.1, 0.3, 0.6]:
        params.append({'Noise':0, 'BlurExtent':0, 'ByteError':ByteError})

    for name in files:
        filename = files[name]
        for param in params:
            [Noise, BlurExtent, ByteError] = [param['Noise'], param['BlurExtent'], param['ByteError']]
            foutname = "Scores_%s_%g_%i_%g.mat"%(name, Noise, BlurExtent, ByteError)
            if not os.path.exists(foutname):
                (PScores, QPScores) = runExperiments(filename, BlockLen, BlockHop, win, dim, NRandDraws, Noise, BlurExtent, ByteError)
                sio.savemat(foutname, {"PScores":PScores, "QPScores":QPScores})
            else:
                print "Already computed %s, skipping..."%foutname
