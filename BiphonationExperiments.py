from VideoTools import *
from ROCExperiments import *
import cv2

def doSlidingWindowVideo(XOrig, dim, Tau, dT, filePrefix, doDerivative = True):
    X = getPCAVideo(XOrig)
    print("Finished PCA")
    if doDerivative:
        [X, validIdx] = getTimeDerivative(X, 10)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    PDs = doRipsFiltration(XS, 1)

    plt.clf()
    makePlot(XS, PDs[1])
    plt.savefig("%s_Stats.png"%filePrefix)
    return PDs[1]

if __name__ == '__main__2':
    NFrames = 400
    A1 = 10
    T1 = 16
    A2 = 10
    T2 = 16*np.pi/3
    ydim = 100
    (I, IDims) = make2ShakingCircles(NFrames, T1, T2, A1, A2, ydim = ydim)
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 16, 20, "2VibratingCircles")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]

if __name__ == '__main__2':
    (I, IDims) = loadVideoFolder("DoublePendulum")
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 20, 80, "DoublePendulum")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]

if __name__ == '__main__2':
    (I, IDims) = loadVideoFolder("TriplePendulum")
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 20, 80, "TriplePendulum")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]

def getHOG(Vid):
    hog = cv2.HOGDescriptor()
    H = hog.compute(Vid[0])
    IRet = np.zeros((len(Vid), len(H)))
    IRet[0, :] = H.flatten()
    for i in range(1, len(Vid)):
        print "Computing HOG %i of %i"%(i, len(Vid))
        H = hog.compute(Vid[i])
        IRet[i, :] = H.flatten()
    return IRet

if __name__ == "__main__":
    (I, IDims) = loadVideo("VocalCordsGradientFull.ogg")
    I = I[700::, :]
    dim = 20
    Tau = 4
    dT = 0.2
    PD = doSlidingWindowVideo(I, dim, Tau, dT, "VocalCords", doDerivative = True)
    idx = np.argsort(-(PD[:, 1] - PD[:, 0]))
    P = PD[idx, 1] - PD[idx, 0]
    print P[0:2]
