import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from Hodge import *

if __name__ == "__main__":
    foldername = "VideoMix/NumberedVideos"
    X = sio.loadmat("TurkResults/Class.mat")
    R = X['R']
    Y = R[:, 2]
    R = R[:, 0:2]
    W = np.ones(Y.shape)
    (s, I, H) = doHodge(R, W, Y, verbose = True)
    INorm = getWNorm(I, W)
    HNorm = getWNorm(H, W)
    idx = np.argsort(s)
    print(idx)

    #Output results in HTML format in descending order of maximum persistence
    fout = open("%s/ClassRankings.html"%foldername, "w")
    fout.write("<html><body>")
    fout.write("<h1>INorm = %g</h1>"%INorm)
    fout.write("<h1>HNorm = %g</h1>"%HNorm)
    fout.write("<table border = '1'>")
    count = 1
    for i in idx:
        fout.write("<tr><td><h2>%i</h2>%i.ogg<BR><BR>s = <b>%g</b></td>"%(count, i, s[i]))
        fout.write("<td><video controls><source src=\"%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video>"%i)
        fout.write("</tr>\n")
        count += 1
    fout.write("</table></body></html>")
    fout.close()
