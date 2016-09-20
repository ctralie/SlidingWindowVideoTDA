import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import subprocess

if __name__ == '__main__':
    files = os.listdir('labels')
    actions = {}
    for f in files:
        s = "labels/%s"%f
        x = sio.loadmat(s)
        a = x['action'][0]
        if not a in actions:
            actions[a] = []
        prefix = f.split(".mat")[0]
        actions[a].append((prefix, x['nframes']))
    #Pick a random video in each action set
    i = 0
    fout = open("out.html", "w")
    fout.write("<table><tr>")
    for a in actions:
        #Pick out the longest videos
        arr = np.array([x[1] for x in actions[a]])
        prefix = actions[a][np.argmax(arr)][0]
        subprocess.call(["avconv", "-r", "30", "-i", "frames/%s/%s6d.jpg"%(prefix, "%"), "-r", "30", "-b", "30000k", "%i.ogg"%i])
        fout.write("<td><video controls>  <source src=\"%i.ogg\" type=\"video/ogg\">Your browser does not support the video tag.</video></td>"%i)
        i = i + 1
        if i%3 == 0:
            fout.write("</tr><tr>")
    fout.write("</tr></table>")
    fout.close()
