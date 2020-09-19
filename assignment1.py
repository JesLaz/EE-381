import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

def prob1(p):
    N = 10000
    n = np.size(p)
    s = np.zeros((N,1))
    cs = np.cumsum(p)
    cp = np.append(0,cs)
    for i in range(0,N):
        r = random.random()
        for j in range(0,n):
            if r>cp[j] and r<=cp[j+1]:
                    d=j+1
        s[i] = d

    #Plotting
    b = range(1, n + 2)
    sb = np.size(b)
    h1, bin_edges = np.histogram(s, bins=b)
    b1 = bin_edges[0:sb - 1]
    prob = h1 / N
    plt.stem(b1, prob)
    plt.title('PMF of n-sided die')
    plt.xlabel('Number on the face of the die')
    plt.ylabel('Probability')
    plt.xticks(b1)
    plt.show()

#Helper method for problem 2; returns the number of attempts to roll a 7 with two dice
def numOfRolls():
    sum = 0
    iter = 1
    while(sum != 7):
        d1 = random.randint(1,6)
        d2 = random.randint(1,6)
        sum = d1 + d2
        iter += 1
    iter -= 1
    return iter

def prob2(N):
    results = [numOfRolls() for i in range(N)]
    numOfAttempts = sorted(set(results))
    instances = [results.count(j)/N for j in numOfAttempts]

    #Plotting
    plt.stem(numOfAttempts, instances)
    #plt.xlim(right=30)
    plt.title('PMF of rolling sum 7 with two die')
    plt.xlabel('Rolls it takes')
    plt.ylabel('Instances occured')
    plt.show()

#Helper method for problem 3; returns a value of 1 if tossed 50 heads out of 100 flips
def tossCoins():
    #Treating 0 as tails and 1 as heads, 50 heads would represent the integer 50
    if sum(np.random.randint(0,2,100)) == 50:
        return 1
    else:
        return 0

def prob3(N):
    total = 0
    for i in range(N):
        total += tossCoins()
    p = total/N
    print("P = ", p)


def experiment(p,N):
    n=np.size(p)
    s=np.zeros((N,1))
    for i in range(0,N):
        d=prob1(p)
        s[i]=d

def main():
    p = np.array([0.10, 0.15, 0.20, 0.05, 0.30, 0.10, 0.10])
    N=10000
    prob1(p)
    prob2(100000)#With N = 100000
    prob3(100000)#With N = 100000

main()