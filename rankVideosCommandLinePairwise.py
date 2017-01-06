import subprocess
import numpy as np
import scipy.io as sio

VideoDir = "KTH/Subject2"
N = 24
I, J = np.meshgrid(np.arange(N), np.arange(N))
I = I[np.triu_indices(N, 1)]
J = J[np.triu_indices(N, 1)]
np.random.seed(100)
idx = np.random.permutation(len(I))
I = I[idx]
J = J[idx]

R = sio.loadmat("R.mat")
R = R['R']
R = [[R[i, 0], R[i, 1], R[i, 2]] for i in range(R.shape[0])]

for a in range(len(R), len(I)):
    i = I[a]
    j = J[a]
    fout = open("temp.html", "w")
    fout.write("<html><body><table><tr><td><h1>1</h1></td><td><h1>2</h1></td></tr><tr>")
    fout.write("<td><video controls><source src=\"%s/%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video></td>"%(VideoDir, i))
    fout.write("<td><video controls><source src=\"%s/%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video></td>"%(VideoDir, j))
    fout.write("</tr></table></body></html>")
    fout.close()
    subprocess.call(["google-chrome", "temp.html"])
    res = raw_input("Enter which one is greater: ")
    if int(res)-1 == 0:
        res = 1
    else:
        res = -1
    R.append([i, j, res])
    sio.savemat("R.mat", {"R":np.array(R)})
