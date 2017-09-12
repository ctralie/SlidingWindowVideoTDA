'''
Implementing the technique in "A Smarter Way To Find Pitch" by Philip McLeod
and Geoff Wyvill, which is simple, elegant, and effective
'''
import numpy as np
import matplotlib.pyplot as plt


def estimateFundamentalFreq(x, doPlot = False):
    #Step 1: Compute normalized squared difference function
    #Using variable names in the paper
    N = x.size
    W = np.int(N/2)
    t = W
    corr = np.zeros(W)
    #Do brute force f FFT because I'm lazy
    #(fine because signals are small)
    for Tau in np.arange(W):
        xdelay = x[Tau::]
        L = (W - Tau)/2
        m = np.sum(x[int(t-L):int(t+L+1)]**2) + np.sum(xdelay[int(t-L):int(t+L+1)]**2)
        r = np.sum(x[int(t-L):int(t+L+1)]*xdelay[int(t-L):int(t+L+1)])
        corr[Tau] = 2*r/m

    #Step 2: Find the ''key max''
    #Compute zero crossings
    zc = np.zeros(corr.size-1)
    zc[(corr[0:-1] < 0)*(corr[1::] > 0)] = 1
    zc[(corr[0:-1] > 0)*(corr[1::] < 0)] = -1

    #Mark regions which are admissible for key maxes
    #(regions with positive zero crossing to left and negative to right)
    admiss = np.zeros(corr.size)
    admiss[0:-1] = zc
    for i in range(1, corr.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i-1]

    #Find all local maxes
    maxes = np.zeros(corr.size)
    maxes[1:-1] = (np.sign(corr[1:-1] - corr[0:-2])==1)*(np.sign(corr[1:-1] - corr[2::])==1)
    maxidx = np.arange(corr.size)
    maxidx = maxidx[maxes == 1]
    maxTau = 0
    if len(corr[maxidx]) > 0:
        maxTau = maxidx[np.argmax(corr[maxidx])]

    if doPlot:
        plt.subplot(211)
        plt.plot(x)
        plt.title("Original Signal")
        plt.subplot(212)
        plt.plot(corr)
        plt.hold(True)
        plt.plot(admiss*1.05, 'r')
        plt.ylim([-1.1, 1.1])
        plt.scatter(maxidx, corr[maxidx])
        plt.scatter([maxTau], [corr[maxTau]], 100, 'r')
        plt.title("Max Tau = %i, Clarity = %g"%(maxTau, corr[maxTau]))
    return (maxTau, corr)

if __name__ == '__main__':
    T = 60
    NPeriods = 5
    np.random.seed(50)
    t = np.linspace(0, 2*np.pi*NPeriods, T*NPeriods)
    slope = t[1] - t[0]
    t = np.cumsum(np.random.rand(t.size)*2*slope)
    x = np.cos(t) + np.cos(2*t)
    f = estimateFundamentalFreq(x, True)
