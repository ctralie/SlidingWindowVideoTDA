import sys
sys.path.append("../")
from VideoTools import *
import subprocess
import os

def saveVideoID(I, IDims, fileprefix, ID, FrameRate = 30, NumberFrames = 30):
    N = I.shape[0]
    frame = np.array([])
    for i in range(N):
        frame = np.reshape(I[i, :], IDims)
        frame[frame < 0] = 0
        frame[frame > 1] = 1
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), frame)
    for i in range(NumberFrames):
        command = ["convert", "%s%i.png"%(TEMP_STR, N), "-fill", "red", "-pointsize", "60", "-draw", 'text 20,60 %s%.3i%s'%("'", ID, "'"), "%s%i.png"%(TEMP_STR, N+i+1)]
        print command
        subprocess.call(command)
        print N + i + 1
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


BlockLen = 160
dirs = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
np.random.seed(100)
IDs = np.random.permutation(999)

i = 0
for d in dirs:
    Videos = ["%s/%s"%(d, v) for v in os.listdir(d) if v.find("person02") > -1]
    for V in Videos:
        if not V[-4::] == ".avi":
            continue
        print "Saving %s..."%V
        (I, IDims) = loadVideo(V)
        I = I[0:BlockLen+10, :]
        saveVideoID(I, IDims, "Subject2/%i"%i, IDs[i])
        i = i + 1
