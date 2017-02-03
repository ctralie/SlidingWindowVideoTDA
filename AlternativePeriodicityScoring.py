"""
Purpose: To implement some alternative techniques for periodicity quantification
to compare with TDA
"""
import numpy as np
import sys
sys.path.append("GeometricCoverSongs")
sys.path.append("GeometricCoverSongs/SequenceAlignment")
from CSMSSMTools import *
import scipy.stats
import scipy.signal
import scipy.ndimage

def getCutlerDavisFrequencyScore(I, doPlot = False):
    """
    Compute the frequency score suggested by Cutler and Davis, with a slight
    modification using Kurtosis instead of mean versus standard deviation
    :param I: An Nxd matrix representing a video with N frames at a resolution of
        d pixels
    :doPlot: If true, show the SSM and average power spectrum across all columns
    """
    N = I.shape[0]
    (D, _) = getSSM(I, N)
    F = np.zeros(N)
    #For linearly detrending
    A = np.ones((N, 2))
    A[:, 1] = np.arange(N)
    #Compute the power spectrum column by column
    for i in range(N):
        x = D[:, i]
        #Linearly detrend
        mb = np.linalg.lstsq(A, x)[0]
        y = x - A.dot(mb)
        #Apply Hann Window
        y = y*np.hanning(N)
        #Add on power spectrum
        F += np.abs(np.fft.fft(y))**2
    #Compute kurtosis of normalized averaged power spectrum
    F = F/np.sum(F)
    F[0:2] = 0 #Ignore DC component
    F[-1] = 0
    kurt = scipy.stats.kurtosis(F, fisher = False)
    M = np.mean(F)
    S = np.std(F)
    if doPlot:
        plt.subplot(121)
        plt.imshow(D, cmap='afmhot', interpolation = 'none')
        plt.subplot(122)
        plt.plot(F)
        plt.hold(True)
        plt.plot([0, N], [M, M], 'b')
        plt.plot([0, N], [M+2*S, M+2*S])
        plt.title("Kurtosis = %.3g"%kurt)
    return (np.max(F) - M)/S


def checkLattice(Q, JJ, II, L, d, offset, CSmooth, doPlot = False):
    P = np.zeros((len(JJ), 2))
    P[:, 0] = II
    P[:, 1] = JJ
    #Find closest point in P to every point in the lattice Q
    CSM = getCSM(Q, P)
    idx = np.argmin(CSM, 1)
    dist = np.min(CSM, 1)
    #Keep only the points that have a normalized correlation value > 0.25
    J = JJ + offset #Index back into the array C
    I = II + offset
    NPs = np.sum(CSmooth[I, J] > 0.25) #Number of maxes to try to match
    scores = CSmooth[I[idx], J[idx]]
    idx = idx[scores > 0.25]
    dist = dist[scores > 0.25]
    #Keep only the points that are closer than d/2
    idx = idx[dist < d/2.0]
    dist = dist[dist < d/2.0]
    J = J[idx]
    I = I[idx]
    #Now compute error and ratio of matched points
    err = np.sum(dist)
    r1 = float(len(idx))/Q.shape[0] #Number of matched lattice point
    idx = np.unique(idx)
    r2 = float(len(idx))/NPs #Number of unmatched points
    denom = r1*r2
    if denom == 0:
        score = np.inf
    else:
        score = (1+err/r1)*((1.0/denom)**3)

    if doPlot:
        #Figure out extent of image so
        e = (-offset, -offset+CSmooth.shape[0]-1, -offset + CSmooth.shape[1]-1, -offset)
        plt.imshow(CSmooth, extent=e, cmap='afmhot', interpolation = 'nearest')
        plt.hold(True)
        #Draw peaks
        plt.scatter(JJ, II, 20, 'r')
        #Draw lattice points
        plt.scatter(Q[:, 1], Q[:, 0], 20, 'b')
        #Draw matched peaks
        plt.scatter(J - offset, I - offset, 30, 'g')

        plt.title("Err = %.3g, Matched: %.3g\nUnmatched:%.3g, score = %.3g"%(err, r1, r2, score))
        plt.xlim([e[0], e[1]])
        plt.ylim([e[2], e[3]])
        plt.xlabel('X Lag')
        plt.ylabel('Y Lag')
    return (err, r1, score, Q)

def checkSquareLattice(JJ, II, L, d, offset, CSmooth, doPlot = False):
    arr = np.arange(0, L+1, d)
    narr = -arr
    arr = narr.tolist()[::-1] + arr.tolist()[1::]
    X, Y = np.meshgrid(arr, arr)
    Q = np.zeros((X.size, 2))
    Q[:, 0] = Y.flatten()
    Q[:, 1] = X.flatten()
    return checkLattice(Q, JJ, II, L, d, offset, CSmooth, doPlot)

def checkDiamondLattice(JJ, II, L, d, offset, CSmooth, doPlot = False):
    arr = np.arange(0, L+1, 2*d)
    narr = -arr
    arr = narr.tolist()[::-1] + arr.tolist()[1::]
    X, Y = np.meshgrid(arr, arr)
    Q1 = np.zeros((X.size, 2))
    Q1[:, 0] = Y.flatten()
    Q1[:, 1] = X.flatten()
    arr2 = np.arange(d, L+1, 2*d)
    narr = -arr2
    arr2 = narr.tolist()[::-1] + arr2.tolist()
    X, Y = np.meshgrid(arr2, arr2)
    Q2 = np.zeros((X.size, 2))
    Q2[:, 0] = Y.flatten()
    Q2[:, 1] = X.flatten()
    Q = np.concatenate((Q1, Q2), 0)
    return checkLattice(Q, JJ, II, L, d, offset, CSmooth, doPlot)

def correlateSquareSameFFT(M1, M2):
    N = M1.shape[0]
    D1 = np.zeros((N*2, N*2))
    D1[0:N, 0:N] = M1
    D2 = np.zeros((N*2, N*2))
    D2[0:N, 0:N] = M2
    F1 = np.fft.fft2(D1)
    F2 = np.fft.fft2(D2)
    F2 = F1*F2
    return np.abs(np.fft.ifft2(F2))

#Inspired by
#https://mail.scipy.org/pipermail/scipy-dev/2013-December/019498.html
def normautocorr2d(a):
    c = correlateSquareSameFFT(a,np.flipud(np.fliplr(a)))
    shape = a.shape
    a = correlateSquareSameFFT(a**2, np.ones(shape))
    c = c/np.sqrt(a**2)
    return c

def getCutlerDavisLatticeScore(I, doPlot = False):
    N = I.shape[0]
    L = int(N/3)
    (D, _) = getSSM(I, N)

    #Step 1: Do count normalized autocorrelation with FFT zeropadding
    C = normautocorr2d(D)

    #Step 2: Apply Gaussian filter
    [JJ, II] = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    sigma = 1
    G = np.exp(-(II**2 + JJ**2)/(2*sigma**2))
    G = G/np.sum(G) #Normalize so max autocorrelation is still 1 after smoothing
    CSmooth = scipy.signal.correlate2d(C, G, 'valid')

    #Step 3: Do peak picking
    CSmooth = CSmooth[N-L:N+L+1, N-L:N+L+1]
    CSmooth = CSmooth - np.min(CSmooth)
    CSmooth = CSmooth/np.max(CSmooth)
    #CSmooth = CSmooth/np.max(CSmooth)
    M = scipy.ndimage.filters.maximum_filter(CSmooth, size=5)
    [JJ, II] = np.meshgrid(np.arange(M.shape[1]), np.arange(M.shape[0]))
    #Account for gaussian filter width after 'valid' convolution
    offset = L - int(np.ceil(G.shape[0]/2.0))
    JJ = JJ[M == CSmooth] - offset
    II = II[M == CSmooth] - offset

    #Step 4: search over lattices
    minscore = np.inf
    minQ = np.array([[]])
    mind = 2
    for d in range(2, L):
        if doPlot:
            plt.clf()
        (err, ratio, score, Q) = checkDiamondLattice(JJ, II, L, d, offset, CSmooth, doPlot)
        if doPlot:
            plt.savefig("DiamondLattice%i.png"%d, bbox_inches='tight')
        if score < minscore:
            minscore = score
            minQ = Q
            mind = d
        if doPlot:
            plt.clf()
        (err, ratio, score, Q) = checkSquareLattice(JJ, II, L, d, offset, CSmooth, doPlot)
        if doPlot:
            plt.savefig("SquareLattice%i.png"%d, bbox_inches='tight')
        if score < minscore:
            minscore = score
            minQ = Q
            mind = d
    return {'score':minscore, 'D':D, 'Q':minQ, 'd':mind, 'L':L, 'offset':offset, 'JJ':JJ, 'II':II, 'CSmooth':CSmooth}

if __name__ == '__main__':
    np.random.seed(10)
    plt.figure(figsize=(12, 6))
    N = 20
    NPeriods = 10
    t = np.linspace(-1, 1, N+1)[0:N]**3
    t = 0.5*t/max(np.abs(t)) + 0.5
    t = 2*np.pi*t
    #t = np.linspace(0, 5*2*np.pi, N)
    X = np.zeros((N*NPeriods, 2))
    for i in range(NPeriods):
        X[i*N:(i+1)*N, 0] = np.cos(t)
        X[i*N:(i+1)*N, 1] = np.sin(t)
    X = X + 0.2*np.random.randn(N*NPeriods, 2)
    #X = np.random.randn(N*NPeriods, 2)
    r = getCutlerDavisLatticeScore(X)

    #Plot results
    plt.figure(figsize=(18, 4))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], 20)
    plt.title('Time Series')
    plt.subplot(132)
    plt.imshow(r['D'], cmap='afmhot', interpolation = 'nearest')
    plt.title('SSM')
    plt.subplot(133)
    checkLattice(r['Q'], r['JJ'], r['II'], r['L'], r['d'], r['offset'], r['CSmooth'], doPlot = True)
    plt.savefig("Lattice.svg", bbox_inches = 'tight')
