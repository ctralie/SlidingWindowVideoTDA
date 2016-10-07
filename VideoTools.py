#Programmer: Chris Tralie
#Purpose: Some tools that load/save videos in Python by wrapping around
#avconv
import numpy as np
import numpy.linalg as linalg
import time
import os
import subprocess
import matplotlib.image as mpimage
import scipy
import scipy.signal
from SyntheticCurves import * #For motion blur

#Need these for saving 3D video

AVCONV_BIN = 'avconv'
TEMP_STR = "pymeshtempprefix"

#############################################################
####                  VIDEO I/O TOOLS                   #####
#############################################################

#Methods for converting to YCbCr (copied matrices from Matlab)
toNTSC = np.array([[0.2989, 0.5959, 0.2115], [0.587, -0.2744, -0.5229], [0.114, -0.3216, 0.3114]])
fromNTSC = np.linalg.inv(toNTSC)

def rgb2ntsc(F):
    return F.dot(toNTSC.T)

def ntsc2rgb(F):
    return F.dot(fromNTSC.T)

#Input: path: Either a filename or a folder
#Returns: tuple (Video NxP array, dimensions of video)
def loadVideo(path, YCbCr = False):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    #Step 1: Figure out if path is a folder or a filename
    prefix = "%s/"%path
    isFile = False
    if os.path.isfile(path):
        isFile = True
        #If it's a filename, use avconv to split it into temporary frame
        #files and load them in
        prefix = TEMP_STR
        command = [AVCONV_BIN,
                    '-i', path,
                    '-f', 'image2',
                    TEMP_STR + '%d.png']
        subprocess.call(command)
    
    #Step 2: Load in frame by frame  
    #First figure out how many images there are
    #Note: Frames are 1-indexed
    NFrames = 0
    while True:
        filename = "%s%i.png"%(prefix, NFrames+1)
        if os.path.exists(filename):
            NFrames += 1
        else:
            break
    if NFrames == 0:
        print("ERROR: No frames loaded")
        return (None, None)
    F0 = mpimage.imread("%s1.png"%prefix)
    IDims = F0.shape
    #Now load in the video
    I = np.zeros((NFrames, F0.size))
    print "Loading video.",
    for i in range(NFrames):
        if i%20 == 0:
            print ".",
        filename = "%s%i.png"%(prefix, i+1)
        IM = mpimage.imread(filename)
        if YCbCr:
            IM = rgb2ntsc(IM)
        I[i, :] = IM.flatten()
        if isFile:
            #Clean up temporary files
            os.remove(filename)
    print("\nFinished loading %s"%path)
    return (I, IDims)

#Output video
#I: PxN video array, IDims: Dimensions of each frame
def saveVideo(I, IDims, filename, FrameRate = 30, YCbCr = False, Normalize = False):
    #Overwrite by default
    if os.path.exists(filename):
        os.remove(filename)
    N = I.shape[0]
    if YCbCr:
        for i in range(N):
            frame = np.reshape(I[i, :], IDims)
            I[i, :] = ntsc2rgb(frame).flatten()
    if Normalize:
        I = I-np.min(I)
        I = I/np.max(I)
    for i in range(N):
        frame = np.reshape(I[i, :], IDims)
        frame[frame < 0] = 0
        frame[frame > 1] = 1
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), frame)
    if os.path.exists(filename):
        os.remove(filename)
    #Convert to video using avconv
    command = [AVCONV_BIN,
                '-r', "%i"%FrameRate,
                '-i', TEMP_STR + '%d.png', 
                '-r', "%i"%FrameRate,
                '-b', '30000k', 
                filename]
    subprocess.call(command)
    #Clean up
    for i in range(N):
        os.remove("%s%i.png"%(TEMP_STR, i+1))


#############################################################
####        SLIDING WINDOW VIDEO TOOLS, GENERAL         #####
#############################################################
def getPCAVideo(I):
    ICov = I.dot(I.T)
    [lam, V] = linalg.eigh(ICov)
    lam[lam < 0] = 0
    V = V*np.sqrt(lam[None, :])
    return V

def getSlidingWindowVideo(I, dim, Tau, dT):
    N = I.shape[0] #Number of frames
    P = I.shape[1] #Number of pixels (possibly after PCA)
    pix = np.arange(P)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim*P))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        f = scipy.interpolate.interp2d(pix, idx[start:end+1], I[idx[start:end+1], :], kind='linear')
        X[i, :] = f(pix, idxx).flatten()
    return X

def getTimeDerivative(I, Win):
    dw = np.floor(Win/2)
    t = np.arange(-dw, dw+1)
    sigma = 0.4*dw
    xgaussf = t*np.exp(-t**2  / (2*sigma**2))
    #Normalize by L1 norm to control for length of window
    xgaussf = xgaussf/np.sum(np.abs(xgaussf)) 
    xgaussf = xgaussf[:, None]
    IRet = scipy.signal.convolve2d(I, xgaussf, 'valid')
    validIdx = np.arange(dw, I.shape[0]-dw, dtype='int64')
    return [IRet, validIdx]


#############################################################
####            FAST TIME DELAY EMBEDDING, Tau = 1      #####
#############################################################
#Input: I: P x N Video with frames along the columns
#W: Windows
#Ouput: Mu: P x W video with mean frames along the columns
def tde_mean(I, W):
    IOut = np.array(I)
    IOut[IOut > 1] = 1
    IOut[IOut < 0] = 0
    start_time = time.time()
    N = I.shape[1]
    P = I.shape[0]
    Mu = np.zeros((P, W))
    for i in range(W):
        Mu[:, i] = np.mean(I[:, np.arange(N-W+1) + i], 1)
    end_time = time.time()
    print("tde_mean elapsed time ", end_time-start_time, " seconds, I.shape = ", I.shape, ", W = ", W)
    return Mu

#Frames assumed to be in each column
#Stacked frames are also in one column
#The delay frames are in a matrix I call "ID" which is never explicitly
#stored
#Return a tuple of (right hand singular vectors, singular values)
def tde_rightsvd(I, W, Mu):
    start_time = time.time()
    N = I.shape[1] #Number of frames in the video
    
    ## Step 1: Precompute frame and mean correlations
    B = I.T.dot(I);
    MuFlat = Mu.flatten()
    MuFlat = np.reshape(MuFlat, [len(MuFlat), 1])
    MuTMu = MuFlat.T.dot(MuFlat)
    C = Mu.T.dot(I) #A WxN matrix
    
    ## Step 2: Use precomputed information to compute (ID-Mu)^T*(ID-Mu)
    #Compute the ID^TID part
    ND = N-W+1
    IDTID = np.zeros((ND, ND))
    #Use the fact that a delay embedding is just a moving average along
    #all diagonals
    for i in range(N-W+1):
        b = np.diag(B, i)
        b2 = np.cumsum(b)
        bend = b2[W-1:]
        bbegin = np.zeros(len(bend))
        bbegin[1:] = b2[0:len(bend)-1]
        b2 = bend - bbegin
        IDTID[np.arange(len(b2)), i + np.arange(len(b2))] = b2
    IDTID = IDTID + IDTID.T
    np.fill_diagonal(IDTID, 0.5*np.diag(IDTID)) #Main diagonal was counted twice
    
    #Compute the Mu^TID part to subtract off mean
    MuTID = np.zeros((1, ND))
    for i in range(ND):
        MuTID[0, i] = np.sum(np.diag(C, i))
    ATA = IDTID - MuTID
    ATA = ATA - MuTID.T
    ATA = ATA + MuTMu
    #Handle numerical precision issues and keep it symmetric
    ATA = 0.5*(ATA + ATA.T)
    
    ## Step 3: Compute right singular vectors
    [S, Y] = linalg.eigh(ATA)
    idx = np.argsort(-S)
    S[S < 0] = 0 #Numerical precision
    S = np.sqrt(S[idx])
    Y = Y[:, idx]
    end_time = time.time()
    return (Y, S)

#############################################################
####               SYNTHETIC 2D VIDEOS                  #####
#############################################################
def makeDriftingOscillatingSquare(NFrames = 200, NPeriods = 8, driftmag = 0, noiseLevel = 0, sigLevel = 0.4, bgLevel = 0.4, res = 20):
    u = np.linspace(-1, 1, res)
    umask = 0.5

    [X, Y] = np.meshgrid(u, u)
    theta = 2*np.pi/7
    omega = 2*np.pi/(1.5)
    wx = omega*np.cos(theta)
    wy = omega*np.sin(theta)

    ts = np.linspace(0, NPeriods*2*np.pi, NFrames+1)
    ts = ts[0:-1] #Make sure the sampling is incommensurate with the period

    #Slow drift
    drift = np.linspace(0, driftmag, NFrames)
    drift = np.reshape(drift, [drift.size, 1])
    drift = np.concatenate((drift, drift), 1)
    I = np.zeros((X.size*3, NFrames))
    IDims = I.shape
    
    for i in range(NFrames):
        mask = (np.abs(X-drift[i, 0]) < umask)*(abs(Y-drift[i,1]) < umask)
        v = ((wx*(X-drift[i, 0]) + wy*(Y-drift[i, 1]) - ts[i]) % (2*np.pi)) < np.pi
        v = bgLevel + sigLevel*v + noiseLevel*np.random.randn(v.shape[0], v.shape[1])
        #v[v > 1] = 1
        v = v*mask
        v = np.reshape(v, [v.shape[0], v.shape[1], 1])
        v = np.concatenate((v, v, v), 2)
        IDims = v.shape
        I[:, i] = v.flatten()
    return (I, IDims, ts)

def makeSmoothGaussianSinusoid(NFrames = 200, NPeriods = 8, dim = 50):
    IDims = (dim, dim, 3)
    I = np.zeros((NFrames, dim*dim*3))
    ts = np.linspace(0, NPeriods*2*np.pi, NFrames+1)
    ts = ts[0:-1]
    
    [X, Y] = np.meshgrid(np.arange(dim), np.arange(dim))
    G = np.zeros((dim, dim, 3))
    for k in range(3):
        G[:, :, k] = np.exp(-((X-float(dim/2))**2 + (Y-float(dim/2))**2)/(2*(dim/8)**2))
    for i in range(NFrames):
        f = G*np.cos(ts[i])
        I[i, :] = f.flatten()
    I = 0.5*(I + 1)
    return (I, IDims)

#############################################################
####                 OTHER VIDEO TOOLS                  #####
#############################################################

def simulateCameraShake(I, IDims, shakeMag):
    J = np.zeros(I.shape)
    for i in range(J.shape[0]):
        print("Blurring frame %i of %i"%(i, J.shape[0]))
        X = np.reshape(I[i, :], IDims)
        (_, mask) = getRandomMotionBlurMask(shakeMag)
        IBlur = 0*X
        for k in range(X.shape[2]):
            IBlur[:, :, k] = scipy.signal.fftconvolve(X[:, :, k], mask, 'same')
        #IBlur = np.array(IBlur, dtype=np.uint8)
        J[i, :] = IBlur.flatten()
    return J


if __name__ == '__main__':
    (I, IDims) = loadVideo("Videos/pendulum.avi")
    IBlur = simulateCameraShake(I, IDims, 40)
    saveVideo(IBlur, IDims, "out.avi")
