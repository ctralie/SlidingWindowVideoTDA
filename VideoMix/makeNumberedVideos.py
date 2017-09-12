import sys
sys.path.append("../")
from VideoTools import *
import subprocess
import os
import scipy.misc

MAXHEIGHT = 160
MINWIDTH = 120

def saveVideoID(I, IDims, fileprefix, ID, FrameRate = 30, NumberFrames = 30):
    N = I.shape[0]
    print(I.shape)
    if I.shape[0] > FrameRate*5:
        I = I[0:FrameRate*5, :]
        N = I.shape[0]
    frame = np.array([])
    print("IDims = ", IDims)
    for i in range(N):
        frame = np.reshape(I[i, :], IDims)
        frame[frame < 0] = 0
        frame[frame > 1] = 1
        if IDims[0] > MAXHEIGHT:
            fac1 = MAXHEIGHT/float(IDims[0])
            fac2 = MINWIDTH/float(IDims[1])
            fac = max(fac1, fac2)
            if i == 0:
                print("Resizing by %g"%fac)
            frame = scipy.misc.imresize(frame, fac)
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), frame)
    PS = 60
    if frame.shape[1] > MINWIDTH*1.5:
        PS = int(30.0*frame.shape[1]/MINWIDTH)
    for i in range(NumberFrames):
        command = ["convert", "%s%i.png"%(TEMP_STR, N), "-fill", "red", "-pointsize", "%i"%PS, "-draw", 'text 20,60 %s%.3i%s'%("'", ID, "'"), "%s%i.png"%(TEMP_STR, N+i+1)]
        print(command)
        subprocess.call(command)
        print(N + i + 1)
    #Convert to video using avconv
    for t in ["avi", "webm", "ogg"]:
        filename = "%s.%s"%(fileprefix, t)
        #Overwrite by default
        if os.path.exists(filename):
            os.remove(filename)
        command = [AVCONV_BIN,
                    '-r', "%i"%FrameRate,
                    '-i', TEMP_STR + '%d.png',
                    '-r', "%i"%FrameRate,
                    '-b', '30000k',
                    filename]
        subprocess.call(command)
    #Clean up
    for i in range(N+NumberFrames):
       os.remove("%s%i.png"%(TEMP_STR, i+1))


np.random.seed(100)
IDs = np.random.permutation(999)

i = 0
Videos = ["OrigVideos/%s"%v for v in os.listdir("OrigVideos")]
for V in Videos:
    print("Saving %s..."%V)
    (I, IDims) = loadVideo(V)
    saveVideoID(I, IDims, "NumberedVideos/%i"%i, IDs[i])
    i = i + 1
