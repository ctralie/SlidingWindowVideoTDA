from VideoTools import *
from ROCExperiments import *
from TDA import *
import sys
sys.path.append("GeometricCoverSongs")
sys.path.append("GeometricCoverSongs/SequenceAlignment")
from SpectralMethods import getDiffusionMap

def getSSM(X):
    XSqr = np.sum(X**2, 1)
    D = XSqr[:, None] + XSqr[None, :] - 2*X.dot(X.T)
    D[D < 0] = 0 #Numerical precision
    D = np.sqrt(D)
    return D

def doPlot(X):
    #Self-similarity matrix
    D = getSSM(X)

    #PCA
    pca = PCA(n_components = 3)
    Y = pca.fit_transform(X)
    sio.savemat("PCA.mat", {"Y":Y})
    eigs = pca.explained_variance_

    plt.clf()
    plt.subplot(221)
    plt.title("Self-Similarity Image")
    plt.xlabel("Frame Number")
    plt.ylabel("Frame Number")
    plt.imshow(D, cmap='afmhot')


    ax = plt.subplot(223)
    ax.set_title("PCA of Sliding Window Embedding")
    c = plt.get_cmap('spectral')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    ax.scatter(Y[:, 0], Y[:, 1], c = C)
    ax.set_aspect('equal', 'datalim')

def printMaxPersistences(PD, num):
    idx = np.argsort(PD[:, 0] - PD[:, 1])
    P = PD[idx, 1] - PD[idx, 0]
    N = min(num, len(idx))
    print P[0:N]


def doSlidingWindowVideo(XOrig, dim, Tau, dT, filePrefix, diffusionParams = None, derivWin = -1):
    X = getPCAVideo(XOrig)
    print X.shape
    print("Finished PCA")
    if derivWin > 0:
        [X, validIdx] = getTimeDerivative(X, derivWin)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]

    #Compute SSM, with optional diffusion
    D = getSSM(XS)
    if diffusionParams:
        print "Doing diffusion..."
        (Kappa, t) = diffusionParams
        XS = getDiffusionMap(D, Kappa, t)
        D = getSSM(XS)
        print "Finished diffusion"

    print "Getting persistence diagrams, N = %i,..."%D.shape[0]
    PDs2 = doRipsFiltrationDM(D, 2, coeff=2)
    PDs3 = doRipsFiltrationDM(D, 2, coeff=3)
    print "Finish getting persistence diagrams"

    plt.figure(figsize=(10, 9))
    doPlot(XS)

    plt.subplot(222)
    H1 = plotDGM(PDs2[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs2[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    #plt.legend(handles=[H1, H2])
    plt.title("Persistence Diagrams Z2")

    plt.subplot(224)
    H1 = plotDGM(PDs3[1], color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
    plt.hold(True)
    H2 = plotDGM(PDs3[2], color = np.array([0.43, 0.67, 0.27]), marker = 'x', sz = 50, label = 'H2', axcolor = np.array([0.8]*3))
    #plt.legend(handles=[H1, H2])
    plt.title("Persistence Diagrams Z3")
    plt.savefig("%s_Stats.svg"%filePrefix, bbox_inches = 'tight')

    return (PDs2[1], PDs3[1], PDs2[2])

def writeVideo(fout, video):
    fout.write("""
        <video controls width = 500>
          <source src="%s" type="video/ogg">
        Your browser does not support the video tag.
        </video>
        """%video)

def writePeriodicityScores(fout, PScore, PScoreMod, HSubscore, QPScore):
    fout.write("<table>")
    fout.write("<tr><td><h2>Periodicity Score</h2></td><td><h2>%.3g</h2></td></tr>"%PScore)
    fout.write("<tr><td><h2>Modified Periodicity Score</h2></td><td><h2>%.3g</h2></td></tr>"%PScoreMod)
    fout.write("<tr><td><h2>Harmonic Subscore</h2></td><td><h2>%.3g</h2></td></tr>"%HSubscore)
    fout.write("<tr><td><h2>Quasiperiodicity Score</h2></td><td><h2>%.3g</h2></td></tr>"%QPScore)
    fout.write("</table>")

if __name__ == "__main__":
    gradSigma = 1
    Videos = []

    #############################################
    ###     Periodic Videos
    #############################################
    #Christan Herbst Periodic
    Videos.append({'file':'VocalCordsVideos/Phasegram_Periodic.mp4', 'name':'HerbstPeriodic', 'startframe':0, 'endframe':-1, 'dim':70, 'Tau':0.5, 'dT':0.5, 'derivWin':10, 'diffusionParams':None})
    
    #MSU Glottis Normal Periodic
    Videos.append({'file':'VocalCordsVideos/NormalPeriodicCrop.ogg', 'name':'NormalPeriodic', 'startframe':0, 'endframe':-1, 'dim':70, 'Tau':0.5, 'dT':0.5, 'derivWin':10, 'diffusionParams':None})
    
    #############################################
    ###     Quasiperiodic Videos
    #############################################
    #AP Biphonation Juergen Neubauer
    Videos.append({'file':'VocalCordsVideos/APBiphonationCrop.mp4', 'name':'APBiphonation', 'startframe':700, 'endframe':1100, 'dim':40, 'Tau':1, 'dT':0.5, 'derivWin':10, 'diffusionParams':None})

    #AP Biphonation 2 Juergen Neubauer Period is about 10 frames
    Videos.append({'file':'VocalCordsVideos/APBiphonation2.mp4', 'name':'APBiphonation2', 'startframe':0, 'endframe':200, 'dim':40, 'Tau':0.25, 'dT':0.25, 'derivWin':2, 'diffusionParams':None})

    #Period is about 8 frames
    Videos.append({'file':'VocalCordsVideos/ClinicalAsymmetry.mp4', 'name':'ClinicalAsymmetry', 'startframe':0, 'endframe':200, 'dim':32, 'Tau':0.25, 'dT':0.25, 'derivWin':2, 'diffusionParams':None})

    #############################################
    ###     Harmonic Videos
    #############################################
    #MSU Glottis Mucus Biphonation
    Videos.append({'file':'VocalCordsVideos/LTR_ED_MucusBiphonCrop.avi', 'name':'MucusBiphonation', 'startframe':0, 'endframe':-1, 'dim':36, 'Tau':1, 'dT':0.25, 'derivWin':10, 'diffusionParams':None})

    #Christian Herbst Subharmonic
    Videos.append({'file':'VocalCordsVideos/Phasegram_Subharmonic.mp4', 'name':'HerbstSubharmonic', 'startframe':0, 'endframe':-1, 'dim':70, 'Tau':0.5, 'dT':0.5, 'derivWin':10, 'diffusionParams':None})


    #############################################
    ###     Perturbed Videos
    #############################################
    #MSU Glottis Mucus Periodic Perturbed
    Videos.append({'file':'VocalCordsVideos/LTR_BO_MucusPertCrop.avi', 'name':'MucusPerturbedPeriodic', 'startframe':0, 'endframe':-1, 'dim':56, 'Tau':0.25, 'dT':0.25, 'derivWin':10, 'diffusionParams':None})
    
    #Christian Herbst Irregular
    Videos.append({'file':'VocalCordsVideos/Phasegram_Irregular.mp4', 'name':'HerbstIrregular', 'startframe':0, 'endframe':600, 'dim':70, 'Tau':0.5, 'dT':1, 'derivWin':10, 'diffusionParams':None})

    #Rayleigh Benard Raw Embeddings
    #Videos = []
    #Videos.append({'file':'RayleighBenard/g21r4000.mpeg', 'name':'RayleighBenardConvection', 'startframe':0, 'endframe':-1, 'dim':70, 'Tau':0.25, 'dT':1, 'derivWin':10, 'diffusionParams':None})

    #Videos.append({'file':'RayleighBenard/Quasiperiodic.mp4', 'name':'RayleighBenardQuasiperiodic', 'startframe':0, 'endframe':-1, 'dim':70, 'Tau':2, 'dT':2, 'derivWin':10, 'diffusionParams':None})

    foutindex = open("VocalCordsResults/index.html", 'w')
    foutindex.write("<html><body>")
    foutindex.write("<table border = 1 cellpadding = 4>")
    foutindex.write("<tr><td><h2>Video Name</h2></td><td><h2>Periodicity Score</h2></td><td><h2>Modified Periodicity Score</h2></td><td><h2>Harmonic Subscore</h2></td><td><h2>Quasiperiodicity Score</h2></td><td><h2>Persistence Diagrams</h2></td></tr>")
    for V in Videos:
        (dim, Tau, dT, name, diffusionParams, derivWin) = (V['dim'], V['Tau'], V['dT'], V['name'], V['diffusionParams'], V['derivWin'])
        i1 = V['startframe']
        i2 = V['endframe']
        (I, IDims) = loadVideo(V['file'])
        I = I[i1:i2, :]
        print "I.shape = ", I.shape
        IGrad = getGradientVideo(I, IDims, gradSigma)
        fout = open("VocalCordsResults/%s.html"%name, 'w')
        fout.write("<html><body")
        fout.write("<pre><h2>dim = %i\nTau = %g\ndT=%g\ndiffusionParams=%s</h2></pre>"%(dim, Tau, dT, diffusionParams))

        saveVideo(I, IDims, "VocalCordsResults/%s.ogg"%name)
        saveVideo(IGrad/np.max(IGrad), IDims, "VocalCordsResults/%sGrad.ogg"%name)

        #Do straight up video
        (I1Z2, I1Z3, I2) = doSlidingWindowVideo(I, dim, Tau, dT, "VocalCordsResults/%s"%name, diffusionParams, derivWin)
        fout.write("<BR><BR><h1>%s</h1><BR>"%name)
        writeVideo(fout, "%s.ogg"%name)
        (PScore, PScoreMod, HSubscore, QPScore) = getPeriodicityScores(I1Z2, I1Z3, I2)
        fout.write("<BR><h2>Scores</h2><BR>")
        writePeriodicityScores(fout, PScore, PScoreMod, HSubscore, QPScore)
        fout.write("<BR><img src = %s_Stats.svg>"%name)
        foutindex.write("<tr><td><a href = %s.html>%s</a></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><img src = %s_Stats.svg width = 200></tr>\n"%(name, name, PScore, PScoreMod, HSubscore, QPScore, name))

        fout.write("<BR><BR><HR>")

        #Do gradient video
        (I1Z2, I1Z3, I2) = doSlidingWindowVideo(IGrad, dim, Tau, dT, "VocalCordsResults/%sGrad"%name, diffusionParams, derivWin)
        fout.write("<BR><BR><h1><a name = \"Grad\">%s Dirichlet Seminorm</a></h1><BR>"%name)
        writeVideo(fout, "%sGrad.ogg"%name)
        (PScore, PScoreMod, HSubscore, QPScore) = getPeriodicityScores(I1Z2, I1Z3, I2)
        fout.write("<BR><h2>Scores</h2><BR>")
        writePeriodicityScores(fout, PScore, PScoreMod, HSubscore, QPScore)
        fout.write("<BR><img src = %sGrad_Stats.svg>"%name)
        foutindex.write("<tr><td><a href = %s.html#Grad>%s Dirichlet Seminorm</a></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><h3>%.3g</h3></td><td><img src = %sGrad_Stats.svg width = 200></tr>\n"%(name, name, PScore, PScoreMod, HSubscore, QPScore, name))

        fout.write("</body></html>")
        fout.close()

    foutindex.write("</table></body></html>")
    foutindex.close()
