"""
Purpose: To implement some alternative techniques for periodicity quantification
to compare with TDA
"""
import numpy as np
import scipy.io as sio
import sys
from CSMSSMTools import *
from VideoTools import *
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

def getD2ChiSquareScore(I, win, dim, derivWin = -1, NBins = 50):
    print("Doing PCA...")
    X = getPCAVideo(I)
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    Tau = win/float(dim-1)
    N = X.shape[0]
    dT = (N-dim*Tau)/float(N)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 0)[None, :]
    #XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    D = getCSM(XS, XS)
    D = D/np.max(D)
    N = D.shape[0]

    #Compute target distribution
    #TODO: Closed form equation for this
    M = N*10
    X = np.zeros((M, 2))
    X[:, 0] = 0.5*np.cos(2*np.pi*np.arange(M)/M)
    X[:, 1] = 0.5*np.sin(2*np.pi*np.arange(M)/M)
    DGT = getCSM(X, X)
    [I, J] = np.meshgrid(np.arange(M), np.arange(M))
    (hGT, edges) = np.histogram(DGT[I > J], bins=50)
    hGT = 1.0*hGT/np.sum(hGT)

    #Compute this distribution
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    (h, edges) = np.histogram(D[I > J], bins = 50)
    h = 1.0*h/np.sum(h)

    #Compute chi squared distance
    num = (h - hGT)**2
    denom = h + hGT
    num[denom <= 0] = 0
    denom[denom <= 0] = 1
    d = np.sum(num / denom)

    return {'score':d, 'h':h, 'hGT':hGT, 'DGT':DGT, 'D':D}

def getDelaunayAreaScore(I, win, dim, derivWin = -1, doPlot = False):
    from SpectralMethods import getDiffusionMap
    from scipy.spatial import Delaunay
    from GeometryTools import getMeanShiftKNN
    print("Doing PCA...")
    X = getPCAVideo(I)
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    Tau = win/float(dim-1)
    N = X.shape[0]
    dT = (N-dim*Tau)/float(N)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 0)[None, :]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    D = getCSM(XS, XS)

    tic = time.time()
    Y = getDiffusionMap(D, 0.1)

    X = Y[:, [-2, -3]]
    XMags = np.sqrt(np.sum(X**2, 1))
    X = X/np.max(XMags)
    X = getMeanShiftKNN(X, int(0.1*N))
    tri = Delaunay(X)

    #Compute all triangle circumcenters
    P0 = X[tri.simplices[:, 0], :]
    P1 = X[tri.simplices[:, 1], :]
    P2 = X[tri.simplices[:, 2], :]
    V1 = P1 - P0
    V2 = P2 - P0
    Bx = V1[:, 0]
    By = V1[:, 1]
    Cx = V2[:, 0]
    Cy = V2[:, 1]
    Dp = 2*(Bx*Cy - By*Cx)
    Ux = (Cy*(Bx**2+By**2)-By*(Cx**2+Cy**2))/Dp
    Uy = (Bx*(Cx**2+Cy**2)-Cx*(Bx**2+By**2))/Dp
    Rs = np.sqrt(Ux**2 + Uy**2) #Radii of points
    Cs = np.zeros((len(Ux), 2))
    Cs[:, 0] = Ux
    Cs[:, 1] = Uy
    Cs = Cs + P0 #Add back offset
    #Prune down to triangle circumcenters which are inside
    #the convex hull of the points
    idx = np.arange(Cs.shape[0])
    idx = idx[tri.find_simplex(Cs) > -1]
    #Find the maximum radius empty circle inside of the convex hull
    [R, cx, cy] = [0]*3
    if len(idx) > 0:
        idxmax = idx[np.argmax(Rs[idx])]
        cx = Ux[idxmax] + P0[idxmax, 0]
        cy = Uy[idxmax] + P0[idxmax, 1]
        R = Rs[idxmax]
    toc = time.time()
    print("Elapsed Time: %g"%(toc-tic))

    if doPlot:
        plt.subplot(131)
        plt.imshow(D, cmap = 'afmhot')
        plt.title("SSM")
        plt.subplot(132)
        plt.imshow(Y, aspect = 'auto', cmap = 'afmhot', interpolation = 'nearest')
        plt.title("Diffusion Map")
        plt.subplot(133)
        plt.scatter(X[:, 0], X[:, 1])
        #simplices = tri.simplices.copy()
        #plt.triplot(X[:, 0], X[:, 1], simplices)
        #Plot maximum circle
        t = np.linspace(0, 2*np.pi, 100)
        plt.scatter(cx, cy, 20, 'r')
        plt.plot(cx + R*np.cos(t), cy + R*np.sin(t))
        plt.axis('equal')
        plt.title("R = %g"%R)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

    return R

if __name__ == '__main__':
    np.random.seed(10)
    plt.figure(figsize=(12, 6))
    N = 20
    NPeriods = 20
    t = np.linspace(-1, 1, N+1)[0:N]#**3
    t = 0.5*t/max(np.abs(t)) + 0.5
    t = 2*np.pi*t
    #t = np.linspace(0, 5*2*np.pi, N)
    X = np.zeros((N*NPeriods, 2))
    for i in range(NPeriods):
        X[i*N:(i+1)*N, 0] = np.cos(t) + 2*np.cos(4*t)
        X[i*N:(i+1)*N, 1] = np.sin(t) + 2*np.sin(4*t)
    X = X + 1*np.random.randn(N*NPeriods, 2)
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

    r = getD2ChiSquareScore(X, N, N)
    plt.figure(figsize=(10, 6))
    plt.plot(r['hGT'], 'k')
    plt.plot(r['h'], 'b')
    plt.savefig("D2.svg", bbox_inches = 'tight')


    plt.figure(figsize=(16, 5))
    getDelaunayAreaScore(X, N, N, doPlot = True)
    plt.savefig("Diffusion.svg", bbox_inches = 'tight')
