import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import os, sys
#
def ThreeSidedDie(p):
    N = 10000
    s = np.zeros((N,1))
    n = 3
    p = np.array([0.3, 0.6, 0.1])
    cs = np.cumsum(p)
    cp = np.append(0,cs)
    for j in range(0,N):
        r = random.random()
        for k in range(0, n):
            if r>cp[k] and r<=cp[k+1]:
                d=k+1
        s[j]=d

    b = range(1,n+2)
    sb=np.size(b)
    h1, bin_edges = np.histogram(s, bins = b)
    b1 = bin_edges[0:sb-1]
    #os.close('all')
    prob = h1/N
    plt.stem(b1,prob)
    plt.title('PMF for an unfair 3-sided die')
    plt.xlabel('Number on the face of the die')
    plt.ylabel('Probability')
    plt.xticks(b1)
    plt.show()

def main():
    ThreeSidedDie(5)

