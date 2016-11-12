from VideoTools import *
from ROCExperiments import *

if __name__ == '__main__2':
    NFrames = 400
    A1 = 10
    T1 = 16
    A2 = 10
    T2 = 16*np.pi/3
    ydim = 100
    (I, IDims) = make2ShakingCircles(NFrames, T1, T2, A1, A2, ydim = ydim)
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 16, 20, "2VibratingCircles")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]

if __name__ == '__main__':
    (I, IDims) = loadVideoFolder("DoublePendulum")
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 40, 40, "DoublePendulum")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]
