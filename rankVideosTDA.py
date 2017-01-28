import numpy as np
from ROCExperiments import *
from VideoTools import *
import scipy.io as sio

if __name__ == "__main__":
    scores = []
    BlockLen = 160
    BlockHop = 80
    win = 20
    dim = 20
    coeff = 2
    foldername = "VideoMix/NumberedVideos"
    NVideos = 20
    for i in range(NVideos):
        (XOrig, FrameDims) = loadVideo("%s/%i.ogg"%(foldername, i))
        XOrig = XOrig[0:-30, :] #Cut out number at the end
        (_, _, s, _, _) = processVideo(XOrig, FrameDims, -1, BlockHop, win, dim, "%s/%iResults"%(foldername, i), coeff = coeff)
        scores.append(s)
    scores = np.array(scores)

    #Output results in HTML format in descending order of maximum persistence
    fout = open("%s/index.html"%foldername, "w")
    fout.write("<html><body><table border = '1'>")
    idx = np.argsort(-scores)
    count = 1
    for i in idx:
        fout.write("<tr><td><h2>%i</h2>%i.ogg<BR><BR>Maximum Persistence = <BR><b>%g</b></td>"%(count, i, scores[i]))
        fout.write("<td><video controls><source src=\"%iResults_max.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video>"%i)
        fout.write("<td><img src = \"%iResults_Stats.png\"></td>"%i)
        fout.write("</tr>\n")
        count += 1
    fout.write("</table></body></html>")
    fout.close()
    print idx

    R = []
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            rel = 1
            if idx[i] < idx[j]:
                R.append([idx[i], idx[j], 1])
            else:
                R.append([idx[j], idx[i], -1])
    sio.savemat("%s/TDARankings.mat"%foldername, {"R":np.array(R)})
