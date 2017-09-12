import numpy as np
from ROCExperiments import *
from VideoTools import *
from AlternativePeriodicityScoring import *
import sys
from SpectralMethods import *
from CSMSSMTools import *
from FundamentalFreq import *
import scipy.io as sio

def getClarity(X):
    #Do Diffusion Maps
    XD = getPCAVideo(X)
    print("Finished PCA")
    [XD, validIdx] = getTimeDerivative(XD, 10)
    D = getCSM(XD, XD)
    XDiffused = getDiffusionMap(D, 0.1)
    x = XDiffused[:, -2] #Get the mode corresponding to the largest eigenvalue
    x = x - np.mean(x)
    (maxT, corr) = estimateFundamentalFreq(x, True)
    return corr

def saveRankings(idx, filename):
    R = []
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            rel = 1
            if idx[i] < idx[j]:
                R.append([idx[i], idx[j], 1])
            else:
                R.append([idx[j], idx[i], -1])
    sio.savemat(filename, {"R":np.array(R)})

if __name__ == "__main__":
    scores = []
    scoresCD = []
    clarity = []
    scoresD2 = []
    BlockLen = 160
    BlockHop = 80
    win = 20
    dim = 20
    foldername = "VideoMix/NumberedVideos"
    NVideos = 20
    for i in range(NVideos):
        (XOrig, FrameDims) = loadVideo("%s/%i.ogg"%(foldername, i))
        XOrig = XOrig[0:-30, :] #Cut out number at the end
        (PScores, MPScores, QPScores, LScores) = processVideo(XOrig, FrameDims, -1, BlockHop, win, dim, "%s/%iResults"%(foldername, i))
        scores.append(PScores[0])
        r = getCutlerDavisLatticeScore(XOrig)
        plt.clf()
        clarity.append(getClarity(XOrig))
        plt.savefig("%s/%i_Clarity.svg"%(foldername, i), bbox_inches='tight')
        s = r['score']
        plt.clf()
        plt.subplot(121)
        plt.imshow(r['D'], cmap='afmhot', interpolation = 'nearest')
        plt.title('SSM')
        plt.subplot(122)
        checkLattice(r['Q'], r['JJ'], r['II'], r['L'], r['d'], r['offset'], r['CSmooth'], doPlot = True)
        plt.savefig("%s/%i_StatsCD.svg"%(foldername, i), bbox_inches='tight')
        scoresCD.append(s)
        
        r = getD2ChiSquareScore(XOrig, 20, 20, derivWin = 10)
        plt.clf()
        plt.plot(r['hGT'], 'k')
        plt.plot(r['h'], 'b')
        plt.savefig("%s/%i_D2Hist.svg"%(foldername, i), bbox_inches='tight')
        scoresD2.append(r['score'])
        
    scores = np.array(scores)
    scoresCD = np.array(scoresCD)
    clarity = np.array(clarity)
    scoresD2 = np.array(scoresD2)

    #Output results in HTML format in descending order of maximum persistence
    fout = open("%s/TDAResults.html"%foldername, "w")
    fout.write("<html><body><table border = '1'>")
    idx = np.argsort(-scores)
    idx2 = np.argsort(scoresCD)
    #idx3 = np.argsort(-clarity)
    idx4 = np.argsort(scoresD2)
    count = 1
    for i in idx:
        fout.write("<tr><td><h2>%i</h2>%i.ogg<BR><BR>Maximum Persistence = <BR><b>%g</b><BR><BR>Kurtosis = <BR><b>%g</b><BR>D2 Dist = <b>%g</b></td>"%(count, i, scores[i], scoresCD[i], scoresD2[i]))
        fout.write("<td><video controls><source src=\"%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video>"%i)
        fout.write("<td><img src = \"%iResults_Stats.svg\"></td>"%i)
        fout.write("<td><img src = \"%i_StatsCD.svg\"></td>"%i)
        fout.write("<td><img src = \"%i_Clarity.svg\"></td>"%i)
        fout.write("<td><img src = \"%i_D2Hist.svg\"></td>"%i)
        fout.write("</tr>\n")
        count += 1
    fout.write("</table></body></html>")
    fout.close()
    print(idx)
    print(idx4)

    saveRankings(idx, "%s/TDARankings.mat"%foldername)
    saveRankings(idx2, "%s/CutlerDavisRankings.mat"%foldername)
    #saveRankings(idx3, "%s/ClarityRankings.mat"%foldername)
    saveRankings(idx4, "%s/D2HistRankings.mat"%foldername)
