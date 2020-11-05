#Jessie Lazo
#EE 381 Section 06

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

#Section 1.1 - Simulate continous random varibles with selected distributions
a = 1.0
b = 4.0
n = 10000

#Generate the values of the RV X via the uniform function
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
print("Standard Deviation\n     Theoretical: ", sig_x_Theoretical, "\n     Experimental: ", sig_x)

#Section 1.2 - Simulate an Exponentially Distributed Random Variable
a = 1
b = 200
n = 10000
beta = 40

#Generate the values of the RV x via the exponential function
x = np.random.exponential(beta, n)

#Create bins and histogram
nbins = 30 # Number of bins
edgecolor = 'w' #color separating bars
bins = [float(x) for x in np.linspace(a,b,nbins+1)]
# bins = [float(x) for x in np.linspace(a,b,nbins+1)]  # Make linspace for 1,400
h1, bin_edges = np.histogram(x,bins,density=True)
# Define points on the horizontal axis
be1 = bin_edges[0:np.size(bin_edges)-1]
be2 = bin_edges[1:np.size(bin_edges)]
b1=(be1+be2)/2
barwidth=b1[1]-b1[0] #width of bars in bargraphs

# Plot the bar graph
fig1 = plt.figure(1)
plt.bar(b1,h1,width=barwidth, edgecolor=edgecolor)

# Plot the exponential pdf
def ExponentialPDF(beta,x):
    f=(1/beta) * np.exp((-1 / beta) * x)* np.ones(np.size(x))
    return f
f = ExponentialPDF(beta,b1)
plt.plot(b1,f,'r')

plt.title('Exponential Distribution')
plt.xlabel('Random Variable')
plt.ylabel('Probability')
plt.show()

#Calculate the theoretical and measured mean and standard deviation
mu_x = np.mean(x)
sig_x = np.std(x)
mu_x_Theoretical = beta
sig_x_Theoretical = beta

print("Section 1.2 - Exponential Random Variable Calculations")
print("Expectation\n     Theoretical: ", mu_x_Theoretical, "\n     Experimental: ", mu_x)
print("Standard Deviation\n     Theoretical: ", sig_x_Theoretical, "\n     Experimental: ", sig_x)

#Section 1.3 - Simulate a Normal Random Variable
#Reset previous variables
a = 0
b = 5
mu = 2.5
sigma = 0.75
n = 10000

#Generate the values of the Normal Variable x via the normal function
x = np.random.normal(mu, sigma, b)

# Create bins and histogram
nbins = 30 # Number of bins
edgecolor = 'w';  #color separating bars
bins = [float(x) for x in np.linspace(a,b,nbins+1)]
# bins = [float(x) for x in np.linspace(a,b,nbins+1)]  # Make linspace for 1,400
h1, bin_edges = np.histogram(x,bins,density=True)
# Define points on the horizontal axis
be1 = bin_edges[0:np.size(bin_edges)-1]
be2 = bin_edges[1:np.size(bin_edges)]
b1=(be1+be2)/2
barwidth=b1[1]-b1[0] #width of bars in bargraphs

# Plot the bar graph
fig1 = plt.figure(1)
plt.bar(b1,h1,width=barwidth, edgecolor=edgecolor)

# Plot the exponential pdf
def NormalPDF(mu, sigma,x):
    f=(   (1/(sigma * math.sqrt(2 * math.pi))) * np.exp((-1 * ((x - mu)**2)) / (2 * (sigma**2))    )           * np.ones(np.size(x)))
    return f
f = NormalPDF(mu,sigma,b1)
plt.plot(b1,f,'r')

plt.title('Normal Distribution')
plt.xlabel('Random Variable')
plt.ylabel('Probability')
plt.show()

#Calculate the theoretical and measured mean and standard deviation
mu_x = np.mean(x)
sig_x = np.std(x)
mu_x_Theoretical = mu
sig_x_Theoretical = sigma

print("Section 1.3 - Normal Random Variable Calculations")
print("Expectation\n     Theoretical: ", mu_x_Theoretical, "\n     Experimental: ", mu_x)
print("Standard Deviation\n     Theoretical: ", sig_x_Theoretical, "\n     Experimental: ", sig_x)

#Section 2 - Central Limit Theorem
def section2(numBooks): #number of books to be 1, 5, 10, and 15
    #Generate the values of the RV X
    a = 1
    b = 4.0
    N = 10000
    mu_x = (a + b) / 2
    sig_x = np.sqrt((b-1)**2 / 12)
    X = np.zeros((N,1))
    for k in range(0, N):
        x = np.random.uniform(a, b, numBooks)
        w = np.sum(x)
        X[k] = w
    #Create bins and histogram
    nbins = 30 #Number of bins
    edgecolor='w'; #Color seperating bars in bargraph
    bins = [float(x) for x in np.linspace(numBooks * a, numBooks * b, nbins + 1)]
    h1, bin_edges = np.histogram(X, bins, density=True)
    # Define points on horizontal axis
    be1 = bin_edges[0:np.size(bin_edges) - 1]
    be2 = bin_edges[1:np.size(bin_edges)]
    b1 = (be1 + be2) / 2
    barwidth = b1[1] - b1[0]  # Width of bars in graph
    plt.close('all')

    #Plot the bar graph
    fig1 = plt.figure(1)
    plt.bar(b1, h1, width=barwidth, edgecolor=edgecolor)

    #Plot the Gaussian Function
    def gaussian(mu, sig, z):
        f = np.exp(-(z-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        return f

    f = gaussian(mu_x*numBooks, sig_x*np.sqrt(numBooks), b1)
    plt.plot(b1, f, 'r')
    plt.title("Gaussian Distribution - Number of Books={}".format(numBooks))
    plt.xlabel("Random Variable")
    plt.ylabel("Probability")
    plt.show()

    #Calculate the theoretical and measured mean and standard deviation
    mu_x = (a+b)/2 * numBooks
    sig_x = np.sqrt((b-a)**2 / 12)

    print("     Mean Thickness", mu_x)
    print("     Standard Deviation of the Thickness", sig_x)

#section2(1)
#section2(5)
#section2(10)
#section2(15)

#Section 3 - Distribution of the Sum of Exponential RVs
a = 1; b = 2000; beta = 40; N = 10000
n=24
carton = []
cartonSum = []

for i in range(N):
    carton = np.random.exponential(beta, n)

    C = (sum(carton)) # b = sum(a)
    cartonSum.append(C)


# Calculate average and standard deviation
mu_c = 24 * beta
sig_c = beta * math.sqrt(24)

# Create bins and histogram
nbins=30; # Number of bins
edgecolor='w'; # Color separating bars in the bargraph
#
bins=[float(carton) for carton in np.linspace(a, b,nbins+1)] # ISSUE: Should I have a and b for this problem?
h1, bin_edges = np.histogram(cartonSum,nbins,density=True) # HAD TO ADD cartonSums as a paremeter

# Define points on the horizontal axis
be1=bin_edges[0:np.size(bin_edges)-1]
be2=bin_edges[1:np.size(bin_edges)]
b1=(be1+be2)/2
barwidth=b1[1]-b1[0] # Width of bars in the bargraphz
plt.close('all')


# Plot bar graph
fig1=plt.figure(1)
plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)

def NormPDF(mu, sigma,x):
    f=((1/(sigma*math.sqrt(2*math.pi))*np.exp((-1*((x-mu)**2))/(2*(sigma**2)))*np.ones(np.size(x))))
    return f


# Plot PDF
f = NormPDF(mu_c, sig_c, b1)
plt.plot(b1,f,'r')
plt.title("PDF of Exponential RV's")
plt.xlabel('Lifetime of Battery in Days - T')
plt.ylabel('PDF')
plt.show()

# Plot bar graph
fig2=plt.figure(2)

def CDF(carton, mu, sigma, x):
    PDF = NormPDF(mu, sigma, x)
    CDF = np.cumsum(barwidth * PDF)
    return CDF

h1 = np.cumsum(h1 * barwidth)
f = CDF(carton, mu_c, sig_c, b1)
plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)
plt.plot(b1, f, 'r')
plt.title("CDF of the Sum of Exponential RV's")
plt.xlabel('Cumulative Sums of Battery Lifetime in Days - T')
plt.ylabel('CDF')
plt.show()
