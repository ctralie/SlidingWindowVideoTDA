import numpy as np
from ROCExperiments import *
from VideoTools import *

if __name__ == "__main__":
    scores = []
    BlockLen = 160
    BlockHop = 80
    win = 20
    dim = 20
    foldername = "KTH/Subject2"
    NVideos = 24
    for i in range(NVideos):
        (XOrig, FrameDims) = loadVideo("%s/%i.ogg"%(foldername, i))
        (_, _, s, _, _) = processVideo(XOrig, FrameDims, BlockLen, BlockHop, win, dim, "%s/%iResults"%(foldername, i))
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
