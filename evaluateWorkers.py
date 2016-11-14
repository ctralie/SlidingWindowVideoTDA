import numpy as np
import matplotlib.pyplot as plt

TDARankings = np.array([int(x) for x in "7 11  4  8 10  2  0  3  6  9 19  5 15 20 13 22  1 23 12 17 16 14 18 21".split()])

#Figure out the TDA ranking for a pair of indices
def getTDARankingDiff(i1, i2):
    idx1 = -1
    idx2 = -1
    for i in range(len(TDARankings)):
        if TDARankings[i] == i1:
            idx1 = i
        elif TDARankings[i] == i2:
            idx2 = i
    if idx1 == idx2:
        print "Error"
    return idx2 - idx1

def writeRow(w, C, I, idx, fout):
    fout.write("\n<tr><td><h3>%s</h3></td><td><h3><center>%i</center></h3></td><td><h3><center>%i</center></h3></td><td><h3>%g %%</h3></td>"%(w, C, I, 100.0*C/(C+I)))
    fout.write("<td>")
    for i in range(len(idx)):
        fout.write("%s<BR>\n"%idx[i])
    fout.write("</td>")
    fout.write("<td><img src = \"%s.svg\"></td></tr>"%w)

def printWorkerWrongRankings(lines, worker):
    for L in lines:
        fields = L.split(",")
        if not (fields[0] == worker):
            continue
        i1 = int(fields[1])
        i2 = int(fields[2])
        ranking = int(fields[3])
        if not (getTDARanking(i1, i2) == ranking):
            print "%i %i"%(i1, i2)

if __name__ == '__main__':
    CORRECT = 0
    INCORRECT = 1
    INCORRECTIDX = 2
    fin = open("Subject2.csv")
    lines = fin.readlines()
    lines = lines[1::]
    workers = {}
    for L in lines:
        fields = L.split(",")
        worker = fields[0]
        i1 = int(fields[1])
        i2 = int(fields[2])
        ranking = int(fields[3])
        if not worker in workers:
            #2-length: First element is number correct
            #second element is an array storing the discrepancy for flipped entries
            workers[worker] = [0, [], []]
        diff = getTDARankingDiff(i1, i2)
        if np.sign(diff) == ranking:
            workers[worker][CORRECT] += 1
        else:
            workers[worker][INCORRECT].append(diff)
            workers[worker][INCORRECTIDX].append([i1, i2])
    TotalCorrect = 0
    TotalIncorrect = 0
    fout = open("Workers/index.html", "w")
    fout.write("<html><body><table cellpadding = 5><tr><td><h2>Worker ID</h2></td><td><h2>Number Correct</h2></td><td><h2>Number Incorrect</h2></td><td><h2>Percentage Correct</h2></td><td><h2>Incorrect Indices</h2></td><td><h2>Incorrect Distance Distribution</h2></td></tr>")
    for w in workers:
        arr = workers[w]
        C = arr[CORRECT]
        I = len(arr[INCORRECT])
        TotalCorrect += C
        TotalIncorrect += I
        plt.clf()
        plt.hist(arr[INCORRECT])
        plt.savefig("Workers/%s.svg"%w)
        writeRow(w, C, I, workers[w][INCORRECTIDX], fout)
    writeRow("Total", TotalCorrect, TotalIncorrect, [], fout)
    fout.write("</table></body></html>")
    fout.close()

    #printWorkerWrongRankings(lines, "A1ZD9SJXQ9C6EW")
