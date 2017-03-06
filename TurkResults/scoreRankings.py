"""Compare all orders pairwise and output results to a table"""
import sys
sys.path.append("..")
from Hodge import *

if __name__ == "__main__":
    RTurk = sio.loadmat("MixRankingsScored.mat")['R']
    RTDAZ2 = sio.loadmat("TDARankingsMixZ2.mat")['R']
    RTDAZ3 = sio.loadmat("TDARankingsMixZ3.mat")['R']
    RCutlerDavisFreq = sio.loadmat("CutlerDavisRankingsFreqMix.mat")['R']
    RCutlerDavisLattice = sio.loadmat("CutlerDavisRankingsLatticeMix.mat")['R']
    RClarity = sio.loadmat("ClarityRankings.mat")['R']
    
    Rs = {'RTurk':RTurk, 'RTDAZ2':RTDAZ2, 'RTDAZ3':RTDAZ3, 'RCutlerDavisFreq':RCutlerDavisFreq, 'RCutlerDavisLattice':RCutlerDavisLattice, 'RClarity':RClarity}
    orders = {}
    fout = open("results.html", "w")
    fout.write("<h1>Orders</h1>\n<table>")
    for RName in Rs:
        R = Rs[RName]
        N = R.shape[0]
        (s, I, H) = doHodge(R[:, 0:2], np.ones(N), R[:, 2].flatten(), verbose = True)
        orders[RName] = np.argsort(s)   
        fout.write("<tr><td>%s</td><td>%s</td></tr>"%(RName, orders[RName]))
    fout.write("</table><BR><BR>")
    
    keys = ['RTDAZ2', 'RTDAZ3', 'RCutlerDavisFreq', 'RCutlerDavisLattice', 'RTurk', 'RClarity']
    
    DsKT = np.zeros((len(Rs), len(Rs)))
    DsJW = np.zeros((len(Rs), len(Rs)))
    for i in range(len(Rs)):
        r1 = orders[keys[i]]
        for j in range(len(Rs)):
            r2 = orders[keys[j]]
            DsKT[i, j] = getKendallTau(r1, r2)
            DsJW[i, j] = getJWDistance(r1, r2)
    
    fout.write("<h1>Kendall Tau</h1>")
    fout.write("<table border = \"1\" cellpadding = \"5\">")
    fout.write("<tr><td></td>")
    for k in keys:
        fout.write("<td>%s</td>"%k)
    fout.write("</tr>")
    for i in range(len(Rs)):
        fout.write("<tr><td>%s</td>"%keys[i])
        for j in range(len(Rs)):
            fout.write("<td>%.3g</td>"%DsKT[i, j])
        fout.write("</tr>\n")
    fout.write("</table>")
    
    
    fout.write("<BR><BR><h1>Jaro Winkler</h1>")
    fout.write("<table border = \"1\" cellpadding = \"5\">")
    fout.write("<tr><td></td>")
    for k in keys:
        fout.write("<td>%s</td>"%k)
    fout.write("</tr>")
    for i in range(len(Rs)):
        fout.write("<tr><td>%s</td>"%keys[i])
        for j in range(len(Rs)):
            fout.write("<td>%.3g</td>"%DsJW[i, j])
        fout.write("</tr>\n")
    fout.write("</table>")
    
