"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To run all vocal folds experiments, including computing persistence
diagrams, periodicity/quasiperiodicity scores, PCA, and periodicity estimation
with diffusion maps + normalized autocorrelation
"""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from VideoTools import *
from ROCExperiments import *
from SpectralMethods import *
from FundamentalFreq import *
from TDA import *
import numpy as np
import scipy.io as sio
import sys

def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return D

def printMaxPersistences(PD, num):
    idx = np.argsort(PD[:, 0] - PD[:, 1])
    P = PD[idx, 1] - PD[idx, 0]
    N = min(num, len(idx))
    print(P[0:N])

def fundamentalFreqEstimation(X):
    #Do Diffusion Maps
    DOrig = getSSM(X)
    XDiffused = getDiffusionMap(DOrig, 0.1)
    x = XDiffused[:, -2] #Get the mode corresponding to the largest eigenvalue
    x = x - np.mean(x)
    (maxT, corr) = estimateFundamentalFreq(x)
    return (x, maxT, corr)

def doSlidingWindowVideo(XOrig, dim, desiredSamples, name, diffusionParams = None, derivWin = -1):
    X = getPCAVideo(XOrig)
    print(X.shape)
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)

    #Do fundmental frequency estimation
    (x, maxT, corr) = fundamentalFreqEstimation(X)
    #Choose sliding window parameters
    Tau = maxT/float(dim)
    #Shoot for a point cloud with about 600 points
    M = X.shape[0] - maxT + 1
    dT = M/float(desiredSamples)
    print("Tau = %g, dT = %g"%(Tau, dT))

    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]

    #Compute SSM
    D = getSSM(XS)

    print("Getting persistence diagrams, N = %i,..."%D.shape[0])
    PDs2 = doRipsFiltrationDM(D, 1, coeff=2)
    PDs3 = doRipsFiltrationDM(D, 2, coeff=3)
    print("Finish getting persistence diagrams")

    plt.clf()
    plt.figure(figsize=(10, 10))


    #PCA
    pca = PCA(n_components = 20)
    Y = pca.fit_transform(XS)
    eigs = pca.explained_variance_ratio_

    plt.clf()
    plt.subplot(221)
    plt.title("Self-Similarity Image")
    plt.xlabel("Frame Number")
    plt.ylabel("Frame Number")
    plt.imshow(D, cmap='afmhot')


    ax = plt.subplot(222)
    ax.set_title("PCA of Sliding Window Embedding\n%.3g%s Variance Explained"%(100*np.sum(eigs[0:2]), "%"))
    c = plt.get_cmap('spectral')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    ax.scatter(Y[:, 0], Y[:, 1], c = C)
    ax.set_aspect('equal', 'datalim')

    plt.subplot(223)
    H1 = plotDGM(PDs2[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs3[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.legend(handles=[H1, H2])
    plt.xlim([0, np.max(PDs3[1])*1.2])
    plt.ylim([0, np.max(PDs3[1])*1.2])
    plt.title("Persistence Diagrams $\mathbb{Z}2$")

    plt.subplot(224)
    H1 = plotDGM(PDs3[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs3[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    plt.legend(handles=[H1, H2])
    plt.xlim([0, np.max(PDs3[1])*1.2])
    plt.ylim([0, np.max(PDs3[1])*1.2])
    plt.title("Persistence Diagrams $\mathbb{Z}3$")

    plt.savefig("VocalCordsResults/%s.svg"%name, bbox_inches='tight')


    plt.clf()
    plt.figure(figsize=(16, 7))
    plt.subplot(211)
    plt.plot(x)
    plt.title("First Diffusion Map Coordinates")
    plt.xlabel("Frame Number")
    plt.subplot(212)
    plt.title("Normalized Squared Difference Autocorrelation (T = %i, Clarity = %.3g)"%(maxT, corr[maxT]))
    plt.plot(corr)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.hold(True)
    plt.scatter([maxT], [corr[maxT]], 100, 'r')
    plt.ylim([-1.1, 1.1])
    plt.xlim([0, len(corr)])
    #plt.subplot(313)
    #plt.plot(Y[:, 0])
    #plt.title("First PC Sliding Window Video")
    plt.savefig("VocalCordsResults/%sAutocorrelation.svg"%name, bbox_inches='tight')

    return (XS, PDs2[1], PDs3[1], PDs3[2], maxT, Tau, dT)

if __name__ == "__main__":
    gradSigma = 1
    dim = 70
    desiredSamples = 600
    Videos = []

    #############################################
    ###     Periodic Videos
    #############################################
    #Christan Herbst Periodic
    Videos.append({'file':'VocalCordsVideos/Phasegram_Periodic.mp4', 'name':'HerbstPeriodic', 'startframe':0, 'endframe':-1, 'derivWin':5, 'diffusionParams':None})

    #MSU Glottis Normal Periodic
    Videos.append({'file':'VocalCordsVideos/NormalPeriodicCrop.ogg', 'name':'NormalPeriodic', 'startframe':0, 'endframe':-1, 'derivWin':5, 'diffusionParams':None})

    #############################################
    ###     Quasiperiodic Videos
    #############################################
    #AP Biphonation Juergen Neubauer
    Videos.append({'file':'VocalCordsVideos/APBiphonationCrop.mp4', 'name':'APBiphonation', 'startframe':700, 'endframe':1100, 'derivWin':5, 'diffusionParams':None})

    #AP Biphonation 2 Juergen Neubauer
    Videos.append({'file':'VocalCordsVideos/APBiphonation2.mp4', 'name':'APBiphonation2', 'startframe':100, 'endframe':350, 'derivWin':5, 'diffusionParams':None})

    #Period is about 8 frames
    Videos.append({'file':'VocalCordsVideos/ClinicalAsymmetry.mp4', 'name':'ClinicalAsymmetry', 'startframe':0, 'endframe':400, 'derivWin':5, 'diffusionParams':None})


    #############################################
    ###     Perturbed Videos
    #############################################
    #MSU Glottis Mucus Periodic Perturbed
    Videos.append({'file':'VocalCordsVideos/LTR_BO_MucusPertCrop.avi', 'name':'MucusPerturbedPeriodic', 'startframe':0, 'endframe':-1, 'derivWin':5, 'diffusionParams':None})

    #Christian Herbst Irregular
    Videos.append({'file':'VocalCordsVideos/Phasegram_Irregular.mp4', 'name':'HerbstIrregular', 'startframe':0, 'endframe':600, 'derivWin':5, 'diffusionParams':None})

    foutindex = open("VocalCordsResults/scores.html", 'w')
    foutindex.write("<html><body>")
    foutindex.write("<table border = 1 cellpadding = 4>")
    foutindex.write("<tr><td><h2>Video Name</h2></td><td><h2>Window Size</h2></td><td><h2>Tau</h2></td><td><h2>dT</h2></td><td><h2>Periodicity Score</h2></td><td><h2>Modified Periodicity Score</h2></td><td><h2>Harmonic Score</h2></td><td><h2>Quasiperiodicity Score</h2></td></tr>")
    for V in Videos:
        (name, diffusionParams, derivWin) = (V['name'], V['diffusionParams'], V['derivWin'])
        i1 = V['startframe']
        i2 = V['endframe']
        (I, IDims) = loadVideo(V['file'])
        I = I[i1:i2, :]
        print("Doing %s..."%name)
        sys.stdout.flush()

        #Do straight up video
        (XS, I1Z2, I1Z3, I2, maxT, Tau, dT) = doSlidingWindowVideo(I, dim, desiredSamples, name, diffusionParams, derivWin)
        (PScore, PScoreMod, HSubscore, QPScore) = getPeriodicityScores(I1Z2, I1Z3, I2)
        foutindex.write("<tr><td>%s</td><td>%.3g</td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td></tr>\n"%(name, maxT, Tau, dT, PScore, PScoreMod, HSubscore, QPScore))
        XS = getPCAVideo(XS)


    foutindex.write("</table></body></html>")
    foutindex.close()
