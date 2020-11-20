#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random

#Section 1 - Effect of Sample Size on Confidence Levels
N = 15000000 #total number of bearings
mu = 55 #population mean
sigma = 5 #population standard deviation
n = range(1, 201) #Sample size; 1, 2, ..., 200
averages = []
pop = np.random.normal(mu, sigma, N)

for i in n:
    X = pop[random.sample(range(N), i)]
    mean = X.mean()
    averages.append(mean)

plt.figure("1A")
plt.title("Sample Means and 95% Confidence Intervals")
plt.xlabel("Sample Size")
plt.ylabel("x_bar")

plt.plot(n, averages, "ob", marker = 'x', linestyle = 'none')
plt.plot(n, [mean for x in n])
X = np.linspace(1, 200)
plt.plot(X, mu + 1.96*sigma / (X**(1/2)), color = 'red', linestyle = '--')
plt.plot(X, mu - 1.96*sigma / (X**(1/2)), color = 'red', linestyle = '--')
plt.ylim(top = 55 + 10)
plt.ylim(bottom = 55 - 10)

plt.figure("1B")
plt.title("Sample Means and 99% Confidence Intervals")
plt.title("Sample means and 99% confidence intervals")
plt.xlabel("Sample Size")
plt.ylabel("x_bar")

plt.plot(n,averages,"ob",marker = 'x', linestyle = 'none')
plt.plot(n,[mean for x in n]) # Plot average line
x = np.linspace(1,200)
plt.plot(X, mu + 2.58*sigma / (x**(1/2)), color = "green", linestyle = '--')
plt.plot(X, mu - 2.58*sigma / (x**(1/2)), color = "green", linestyle = '--')
plt.ylim(top = 55 + 10)
plt.ylim(bottom = 55 - 10)
plt.show()

#Section 2 - Using a Sample Mean to Estimate the Population Mean
