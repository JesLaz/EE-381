#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Section 1 - Simulate continous random varibles with selected distributions
a = 1.0
b = 4.0
n = 10000

#Generate the values of the RV X
x = np.random.uniform(a, b, n)

#Create bins and histograms
nbins = 30 #Number of bins
edgecolor = 'w' #Color seperating bars in the bargraph
bins = [float(x) for x in np.linspace(a, b, nbins + 1)]
h1, bin_edges = np.histogram(x, bins, density=True)
#Define the points on the horizontal azis
be1 = bin_edges[0:np.size(bin_edges)-1]
be2 = bin_edges[1:np.size(bin_edges)]
b1 = (be1 + be2)/2
barwidth = b1[1] = b1[0]

#Plot the bar graph
fig1 = plt.figure(1)
plt.bar(b1, h1, width=barwidth, edgecolor=edgecolor)

#Plot the uniform PDF
def UnifPDF(a, b, x):
    f = (1 / abs(b-a))*np.ones(np.size(x))
    return f
f = UnifPDF(a, b, b1)
plt.plot(b1, f, 'r')

plt.title("Uniform Distribution")
plt.xlabel("Random Variable")
plt.ylabel("Probability")
plt.show()

#Calculate the theoretical and measured mean and standard deviation
mu_x = np.mean(x)
sig_x = np.std(x)
mu_x_Theoretical = a + b/2
sig_x_Theoretical = ((b-a)**2) / 12

print("Section 1.1 - Uniform Random Variable Calculations")
print("Expectation\n     Theoretical: ", mu_x_Theoretical, "\n     Experimental: ", mu_x)
print("Standard Deviation\n     Theoretical: ", sig_x, "\n     Experimental: ", sig_x)
