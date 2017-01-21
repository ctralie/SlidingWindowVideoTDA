"""Compare all orders pairwise and output results to a table"""
import sys
sys.path.append("..")
from Hodge import *

if __name__ == "__main__":
    R21 = sio.loadmat("Subject2Rankings1.mat")['R']
    R22 = sio.loadmat("Subject2Rankings2.mat")['R']
    RMy = sio.loadmat("MyRankings.mat")['R']
    RTDA = sio.loadmat("TDARankings.mat")['R']
    
    Rs = {'R21':R21, 'R22':R22, 'RMy':RMy, 'RTDA':RTDA}
    orders = {}
    fout = open("results.html", "w")
    fout.write("<h1>Orders</h1>\n<table>")
    for RName in Rs:
        R = Rs[RName]
        N = R.shape[0]
        (s, I, H) = doHodge(R[:, 0:2], np.ones(N), R[:, 2].flatten(), verbose = True)
        orders[RName] = np.argsort(s)   
        fout.write("<tr><td>%s</td><td>%s</td></tr>"%(RName, orders[RName]))
        if RName == 'RTDA':
            orders['RTDAReverse'] = np.argsort(-s)  
            fout.write("<tr><td>%s</td><td>%s</td></tr>"%('RTDAReverse', orders['RTDAReverse']))
    fout.write("</table><BR><BR>")
    Rs['RTDAReverse'] = RTDA   
    
    #Flip first and last in RTDA to check stability
    #r = orders['RTDA']
    #f = r[0]
    #l = r[-1]
    #r[0] = l
    #r[-1] = f
    #orders['RTDA'] = r
    
    keys = ['RTDA', 'RTDAReverse', 'RMy', 'R21', 'R22']
    
    DsKT = np.zeros((len(Rs), len(Rs)))
    DsJW = np.zeros((len(Rs), len(Rs)))
    for i in range(len(Rs)):
        r1 = orders[keys[i]]
        for j in range(i+1, len(Rs)):
            r2 = orders[keys[j]]
            DsKT[i, j] = getKendallTau(r1, r2)
            DsJW[i, j] = getJWDistance(r1, r2)
    
    DsKT = DsKT + DsKT.T
    DsJW = DsJW + DsJW.T
    
    fout.write("<h1>Kendall Tau</h1>")
    fout.write("<table border = \"1\" cellpadding = \"5\">")
    fout.write("<tr><td></td>")
    for k in keys:
        fout.write("<td>%s</td>"%k)
    fout.write("</tr>")
    for i in range(len(Rs)):
        fout.write("<tr><td>%s</td>"%keys[i])
        for j in range(len(Rs)):
            fout.write("<td>%g</td>"%DsKT[i, j])
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
            fout.write("<td>%g</td>"%DsJW[i, j])
        fout.write("</tr>\n")
    fout.write("</table>")
    
