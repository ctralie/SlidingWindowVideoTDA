import numpy as np

TDARankings = np.array([int(x) for x in "7 11  4  8 10  2  0  3  6  9 19  5 15 20 13 22  1 23 12 17 16 14 18 21".split()])

#Figure out the TDA ranking for a pair of indices
def getTDARanking(i1, i2):
    for i in TDARankings:
        if i == i1:
            return 1
        elif i == i2:
            return -1
    print "ERROR"
    return 0

def writeRow(w, C, I, fout):
    fout.write("\n<tr><td><h3>%s</h3></td><td><h3><center>%i</center></h3></td><td><h3><center>%i</center></h3></td><td><h3>%g %%</h3></td></tr>"%(w, C, I, 100.0*C/(C+I)))

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
    fin = open("BatchResultsPruned.csv")
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
            #second element is number incorrect
            workers[worker] = [0, 0]
        if getTDARanking(i1, i2) == ranking:
            workers[worker][CORRECT] += 1
        else:
            workers[worker][INCORRECT] += 1
    TotalCorrect = 0
    TotalIncorrect = 0
    fout = open("workers.html", "w")
    fout.write("<html><body><table cellpadding = 5><tr><td><h2>Worker ID</h2></td><td><h2>Number Correct</h2></td><td><h2>Number Incorrect</h2></td><td><h2>Percentage Correct</h2></td></tr>")
    for w in workers:
        arr = workers[w]
        C = arr[CORRECT]
        I = arr[INCORRECT]
        TotalCorrect += C
        TotalIncorrect += I
        writeRow(w, C, I, fout)
    writeRow("Total", TotalCorrect, TotalIncorrect, fout)
    fout.write("</table></body></html>")
    fout.close()

    #printWorkerWrongRankings(lines, "A1ZD9SJXQ9C6EW")
