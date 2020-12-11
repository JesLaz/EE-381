#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

#Section 1 - A Three-State Markov Chain
#Probability Matrix
pA = [1/2, 1/3, 1/4]
pB = [1/4, 1/8, 5/8]
pC = [1/3, 2/3, 3/4]
pI = [1/4, 0, 3/4]

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

def section1():
    plt.figure("Section: 1A")
    plt.title("Three-State Markov Chain")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    x = range(15)
    plt.plot(x, createMarkov(15), "ro", LINESTYLE='--')
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

    plt.figure("Section: 1B")
    plt.title("Three-State Markov Chain: 10000 Trails")
    plt.xlabel("Step Number")
    plt.ylabel("State")
    x = range(15)
    plt.plot(x, stateA, "ro", LINESTYLE='--', label="State 0")
    plt.plot(x, stateB, "bo", LINESTYLE='--', label="State 1")
    plt.plot(x, stateC, "go", LINESTYLE='--', label="State 2")
    plt.legend(loc = "upper right")
    plt.show()

section1()
