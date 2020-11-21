#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random

N = 1500000
mu = 55  # population mean
sigma = 5  # population standard deviation
n = range(1, 201)  # Sample size; 1, 2, ..., 200
M = 10000 # number of trails for section 2
pop = np.random.normal(mu, sigma, N)
b = mu + np.random.randn(N) * sigma

#Section 1 - Effect of Sample Size on Confidence Levels
def section1():
    print("mu: ", mu)
    print("sigma: ", sigma)
    averages = []
    for i in n:
        X = pop[random.sample(range(N), i)]
        mean = X.mean()
        averages.append(mean)

    plt.figure("1A")
    plt.title("Sample Means and 95% Confidence Intervals")
    plt.xlabel("Sample Size")
    plt.ylabel("x_bar")

    plt.plot(n, averages, "ob", marker='x', linestyle='none')
    plt.plot(n, [mean for x in n])
    X = np.linspace(1, 200)
    plt.plot(X, mu + 1.96 * sigma / (X ** (1 / 2)), color='red', linestyle='--')
    plt.plot(X, mu - 1.96 * sigma / (X ** (1 / 2)), color='red', linestyle='--')
    plt.ylim(top=55 + 10)
    plt.ylim(bottom=55 - 10)

    plt.figure("1B")
    plt.title("Sample Means and 99% Confidence Intervals")
    plt.title("Sample means and 99% confidence intervals")
    plt.xlabel("Sample Size")
    plt.ylabel("x_bar")

    plt.plot(n, averages, "ob", marker='x', linestyle='none')
    plt.plot(n, [mean for x in n])  # Plot average line
    x = np.linspace(1, 200)
    plt.plot(X, mu + 2.58 * sigma / (x ** (1 / 2)), color="green", linestyle='--')
    plt.plot(X, mu - 2.58 * sigma / (x ** (1 / 2)), color="green", linestyle='--')
    plt.ylim(top=55 + 10)
    plt.ylim(bottom=55 - 10)
    plt.show()

#Section 2 - Using a Sample Size on Confidence Intervals
def section2(t95, t99, size):
    n95Counter = 0
    n99Counter = 0
    t95Counter = 0
    t99Counter = 0
    s = size
    for i in range(0, M):
        y = b[random.sample(range(N), s)]
        yMean = np.sum(y) / s
        total = 0
        for j in range(0, len(y)):
            total = total + (y[j] - yMean)**2
        yS = total/(s - 1)
        yS = math.sqrt(yS)
        yStandard = yS/math.sqrt(s)

        yUpper95 = yMean + (1.96 * yStandard)
        yLower95 = yMean - (1.96 * yStandard)
        yUpper99 = yMean + (2.58 * yStandard)
        yLower99 = yMean - (2.58 * yStandard)

        tUpper95 = yMean + t95 * (yStandard)
        tLower95 = yMean - t95 * (yStandard)
        tUpper99 = yMean + t99 * (yStandard)
        tLower99 = yMean - t99 * (yStandard)

        if (yLower95 <= mu and yUpper95 >= mu):
            n95Counter += 1
        if (yLower99 <= mu and yUpper99 >= mu):
            n99Counter += 1
        if (tLower95 <= mu and tUpper95 >= mu):
            t95Counter += 1
        if (tLower99 <= mu and tUpper99 >= mu):
            t99Counter += 1

    print("Success Rate of Sample:", size, " With 95% Confidence Interval\n     Normal Distribution:", (n95Counter/M))
    print("Success Rate of Sample:", size, " With 99% Confidence Interval\n     Normal Distribution:", (n99Counter/M))
    print("Success Rate of Sample:", size, " With 95% Confidence Interval\n     Student's Distribution:", (t95Counter/M))
    print("Success Rate of Sample:", size, " With 99% Confidence Interval\n     Student's Distribution:", (t99Counter/M))
    print('')

section1()
section2(2.78, 4.6, 5)
section2(2.02, 2.7, 40)
section2(1.98, 2.62, 120)
