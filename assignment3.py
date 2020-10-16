#Binomiall distribution
#P(X) = (n c x)p^x * q^(n-x)
#n = number of trials, x = number of successful attempts, p = probability, q = probabilty of failure

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

#Probability vector of multi-sided dice given from prompt
pr = [0.2, 0.1, 0.15, 0.3, 0.2, 0.05]
#Number of attempts for experiment 1
n = 1000


def nSidedDice(prb):
    n = len(prb)
    cs = np.cumsum(prb)
    cp = np.append(0, cs)
    r = np.random.random()
    for k in range(0, n):
        if r > cp[k] and r <= cp[k + 1]:
            d = k + 1
            break
    return d

def tripleRoll():
    if(nSidedDice(pr) == 1 and nSidedDice(pr) == 2 and nSidedDice(pr) == 3):
        return True

def successfulAttempts():
    sAttempt = 0
    for i in range(n):
        if(tripleRoll() == True):
            sAttempt += 1
    return sAttempt

#Helper function for binomial distribution
def nChooseR(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def binomial(x):
    #p = Probability of success; in this case its p(Dice 1 == 1, Dice 2 == 2, Dice 3 == 3) = 0.003
    p = 0.003
    q = 1 - p
    result = (nChooseR(1000,x) * (p**x) * (q**(1000-x)))
    return result

def poissonDist(x):
    #p = Probability of success; in this case its p(Dice 1 == 1, Dice 2 == 2, Dice 3 == 3) = 0.003
    p = 0.003
    lam = 1000*p
    result = (lam**x)**(math.e**(-lam)) / (math.factorial(x))
    return result

def experiment1():
    results = [successfulAttempts() for _ in range(10000)]
    xAxis = range(len(set(results)))
    yAxis = [results.count(x)/10000 for x in xAxis]

    plt.stem(xAxis, yAxis, use_line_collection=True)
    plt.title("PMF of Tossing 3 Die")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.xlim(right = 15)
    plt.show()

def experiment2():
    xAxis = range(15)
    yAxis = [binomial(X) for X in xAxis]

    plt.stem(xAxis, yAxis, use_line_collection=True);
    plt.title("PMF of Bernoulli Trail via Binomial Formula");
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.xlim(right=15)
    plt.show()

def experiment3():
    xAxis = range(15)
    yAxis = [poissonDist(X) for X in xAxis]

    plt.stem(xAxis, yAxis, use_line_collection=True)
    plt.title("PMF of Bernoulli Trials via Poisson Distribution")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.xlim(right=15)
    plt.show()

def main():
    #experiment1()
    experiment2()
    experiment3()


startTime = time.time()
main()
print("Runtime: ---%s seconds ---" % (time.time() - startTime))