"""Videos are randomly generated using a Windows executable and stored in
directories before this code is run"""
from VideoTools import *
from TDA import *
from sklearn.decomposition import PCA
import numpy as np
import scipy.io as sio
import os
from ROCExperiments import *

def runExperiments(foldername, packetLoss, NDraws, BlockLen, BlockHop, win, dim, filePrefix):
    print("PROCESSING ", foldername, "  ", packetLoss, "....")
    persistences = []
    for i in range(NDraws):
        print("%i of %i"%(i, NDraws))
        filename = "%s/%i_%i.avi"%(foldername, packetLoss, i)
        if not os.path.exists(filename):
            continue
        (XOrig, FrameDims) = loadVideo(filename)
        thisfilePrefix = filePrefix
        if i > 0:
            thisfilePrefix = ""
        (PDMax, XMax, maxP, maxj, p) = processVideo(XOrig, FrameDims, BlockLen, BlockHop, win, dim, thisfilePrefix)
        persistences = persistences + p
    return np.array(persistences)

if __name__ == '__main__':
    BlockLen = 160
    BlockHop = 20
    win = 20
    dim = 20
    NDraws = 200

    TrueFolder = 'Videos/PendulumEncoded'
    FalseFolder = 'Videos/LandscapeEncoded'
    for packetLoss in range(10):
        if os.path.exists("psTruePacket%i.mat"%packetLoss):
            print("Loading precomputed psTrue packetLoss=%i"%packetLoss)
            psTrue = sio.loadmat("psTruePacket%i.mat"%packetLoss)['psTrue']
        else:
            psTrue = runExperiments(TrueFolder, packetLoss, NDraws, BlockLen, BlockHop, win, dim, "Videos/PendulumPacket%i"%packetLoss)
            sio.savemat("psTruePacket%i.mat"%packetLoss, {"psTrue":psTrue})
        if os.path.exists("psFalsePacket%i.mat"%packetLoss):
            psFalse = sio.loadmat("psFalsePacket%i.mat"%packetLoss)['psFalse']
        else:
            psFalse = runExperiments(FalseFolder, packetLoss, NDraws, BlockLen, BlockHop, win, dim, "Videos/DrivingPacket%i"%packetLoss)
            sio.savemat("psFalsePacket%i.mat"%packetLoss, {"psFalse":psFalse})

        #Plot ROC curve
        (FP, TP) = getROC(psTrue, psFalse)
        plt.clf()
        plt.plot(FP, TP, 'b')
        plt.hold(True)
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'r')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("ROC_PacketLoss%i.svg"%packetLoss, bbox_inches='tight')

        #Plot histogram
        plt.clf()
        plt.hold(True)
        plt.hist(psTrue.flatten(), 10, facecolor='blue')
        plt.hist(psFalse.flatten(), 10, facecolor='red')
        plt.savefig("Hists_PacketLoss%i.svg"%packetLoss, bbox_inches='tight')
