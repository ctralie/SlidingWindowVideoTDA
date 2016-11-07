import subprocess
import numpy as np
import scipy.io as sio

NVideos = 10

R = []
for i in range(NVideos):
    for j in range(i+1, NVideos):
        fout = open("temp.html", "w")
        fout.write("<html><body><table><tr><td><h1>1</h1></td><td><h1>2</h1></td></tr><tr>")
        fout.write("<td><video controls><source src=\"Initial10/%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video></td>"%i)
        fout.write("<td><video controls><source src=\"Initial10/%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video></td>"%j)
        fout.write("</tr></table></body></html>")
        fout.close()
        subprocess.call(["google-chrome", "temp.html"])
        res = raw_input("Enter which one is greater: ")
        if int(res)-1 == 0:
            res = i
        else:
            res = j
        R.append([i, j, res])
        sio.savemat("R.mat", {"R":np.array(R)})
