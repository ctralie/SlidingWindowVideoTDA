from VideoTools import *
from ROCExperiments import *

if __name__ == '__main__':
    NFrames = 400
    T1 = 16
    T2 = 16*np.pi/3
    (I, IDims) = make2GaussianPulses(NFrames, T1, T2)
    (PDMax, XMax, maxP, maxj, persistences) = processVideo(I, IDims, -1, 1, 16, 20, "2Pulses")
    print np.sort(PDMax[:, 1] - PDMax[:, 0])[-2::]
