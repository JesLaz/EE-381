#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

#Section 1 - A Three-State Markov Chain
#Probability Matrix
pA = [1/2, 1/4, 1/4]
pB = [1/4, 1/8, 5/8]
pC = [1/3, 2/3, 3/4]
pI = [1/4, 0, 3/4]

#Probability Matrix for Section 3
#     0    1    2    3    4
p0 = [1,   0,   0,   0,   0]
p1 = [0.3, 0,   0.7, 0,   0]
p2 = [0,   0.5, 0,   0.5, 0]
p3 = [0,   0,   0.6, 0,   0.4]
p4 = [0,   0,   0,   0,   1]

def nSidedDie(p):
    n=np.size(p)
    cs=np.cumsum(p)
    cp=np.append(0,cs)
    r=random.random()
    for k in range(0,n):
        if r>cp[k] and r<=cp[k+1]:
            d=k+1
            break
    return d

def nSidedDieN(p):
    return nSidedDie(p) - 1

def createMarkov(cLen):
    S = [nSidedDieN(pI)]
    for i in range(cLen - 1):
        if(S[-1] == 0):
            S.append(nSidedDieN(pA))
        elif(S[-1] == 1):
            S.append(nSidedDieN(pB))
        elif(S[-1] == 2):
            S.append(nSidedDieN(pC))
    return S

def createMarkov2(cLen):
    S = [2]
    for i in range(cLen - 1):
        if S[-1] == 0:
            S.append(nSidedDieN(p0))
        elif S[-1] == 1:
            S.append(nSidedDieN(p1))
        elif S[-1] == 2:
            S.append(nSidedDieN(p2))
        elif S[-1] == 3:
            S.append(nSidedDieN(p3))
        elif S[-1] == 4:
            S.append(nSidedDieN(p4))
    return S

def section1():
    plt.figure("Figure: 1A")
    plt.title("Three-State Markov Chain")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    x = range(15)
    plt.plot(x, createMarkov(15), "ro", LINESTYLE='-')
    plt.show()
    chain = [(createMarkov(15)) for i in range(10000)]
    transposedChain = np.transpose(chain)
    stateA = []
    stateB = []
    stateC = []

    for r in transposedChain:
        stateA.append((list(r).count(0) / 10000))
        stateB.append((list(r).count(1) / 10000))
        stateC.append((list(r).count(2) / 10000))

    plt.figure("Figure: 1B")
    plt.title("Three-State Markov Chain: 10000 Trails")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    x = range(15)
    plt.plot(x, stateA, "ro", LINESTYLE='-', label="Rain")
    plt.plot(x, stateB, "bo", LINESTYLE='-', label="Snow")
    plt.plot(x, stateC, "go", LINESTYLE='-', label="Ice")
    plt.legend(loc = "upper right")
    plt.show()

    plt.figure("Figure: 1C")
    plt.title("Three-State Markov Chain: Calculated")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    x = range(15)
    plt.plot(x, stateA, "ro", LINESTYLE='-', label="Rain")
    plt.plot(x, stateB, "bo", LINESTYLE='-', label="Snow")
    plt.plot(x, stateC, "go", LINESTYLE='-', label="Ice")
    plt.legend(loc="upper right")
    plt.show()



#Section 2 - The Google PageRank Algorithm
def section2():
    P = np.matrix([[0, 1, 0, 0, 0],
                   [1/2, 0, 1/2, 0, 0],
                   [1/3, 1/3, 0, 0, 1/3],
                   [1, 0, 0, 0, 0],
                   [0, 1/3, 1/3, 1/3, 0]])
    V = [[1/5, 1/5, 1/5, 1/5, 1/5],
         [0,   0,   0,   0,   1]]

    for i in range(2):
        t = np.transpose(V[i])
        nP = V[i]
        bigList = [V[i]]

        for num in range(20):  # 20
            nP = np.matmul(nP, P)
            step = nP.tolist()[0]
            bigList.append(step)

        data = np.transpose(bigList).tolist()
        plt.figure("Figure 2")
        plt.title("Google Page-Rank with Vector: '{}'".format(V[i]))
        plt.xlabel("Chain Position")
        plt.ylabel("State")
        x = range(21)
        plt.plot(x, data[0], 'ro', LINESTYLE='-', label='A')
        plt.plot(x, data[1], 'bo', LINESTYLE='-', label='B')
        plt.plot(x, data[2], 'go', LINESTYLE='-', label='C')
        plt.plot(x, data[3], 'ko', LINESTYLE='-', label='D')
        plt.plot(x, data[4], 'co', LINESTYLE='-', label='E')
        plt.legend(loc='upper right')
        plt.show()

        print("Probabilities For Vector {} In Descending Order".format(V[i]))
        results = [(str(x[-1])[:6], data.index(x)) for x in data]
        printResults = [print("State: {} Probability: {}".format(x[1],x[0])) for x in sorted(results,reverse=True)]




def section3():
    plt.figure("Figure 3")
    plt.title("Five-State Absorbing Markov Chain")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    plt.plot(range(15), createMarkov2(15), "ro", LINESTYLE='-')
    plt.show()


def section4():
    absorbedState = [createMarkov2(15)[-1] for i in range(10000)]
    ended0 = absorbedState.count(0)/10000
    ended4 = absorbedState.count(4)/10000
    print("Absorption Probabilities\n     b20:", ended0, "\n     b24:", ended4)

section1()
section2()
section3()
section4()
