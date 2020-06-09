# SPRT-test

#SPRT test per una popolazione normale con due ipotesi H_0 e H_1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s = np.random.normal(1, 1, 1000)

sigma = 1
x = np.zeros(1000)
#print(x)
p_0 = 1
p_1 = 1
theta_1 = 1
theta_0 = 0

alpha = 0.05
beta = 0.05

A = 0.95/(0.05)
B = 0.05/(0.95)

j=0
i=0

while i<1000:
    x[i] = s[i]
    #print(x[i])
    p_0 = p_0*np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_0)**2))
    p_1 = p_1*np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_1)**2))
    
#    print(p_0)
#    print(p_1)
    if p_1/p_0 <= B:
        print("accept H_0")
        print("The size of the population needed to end the test is: " + str(j))
        i=1000
    elif p_1/p_0 >= A: 
        print("accept H_1")
        print("The size of the population needed to end the test is: " + str(j))
        i=1000
    elif j == 999:
        print("End without response")
        i=1000
    else: 
        #print("repeat")
        i += 1
        j = i + 1









#SPRT test per una popolazione normale con due ipotesi H_0 e H_1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

number = input("How many tests do you want to run? ")
n = int(number)
print(n)
cycle = 0

while cycle < n:
    
    #print(cycle)
    print("This is a SPRT Test for a Random Normal Population (0,1) with two hypothesis H_0 and H_1")
    s = np.random.normal(0, 1, 1000)

    sigma = 1
    x = np.zeros(1000)
    #print(x)
    sum_up = 0
    theta_z = input("Enter the parameter for the Null Hypothesis H_0: ") 
    theta_o = input("Enter the parameter for the Alternative Hypothesis H_1: ") 
    theta_1 = float(theta_o)
    theta_0 = float(theta_z)
    
    alpha = 0.05
    beta = 0.05
    
    error_alpha_in = input("Enter the error of first type: ")
    error_beta_in = input("Enter the error of second type: ")
    
    error_alpha = float(error_alpha_in)
    error_beta = float(error_beta_in)
    
    A = (1 - error_beta)/error_alpha
    B = error_beta/(1 - error_alpha)
    
    j=0
    i=0
    k=1
    while i<1000:
        x[i] = s[i]
        #print(x[i])
        sum_up = sum_up + x[i]
        
        acceptance_number = ((((sigma)**2) / (theta_1 - theta_0))*(np.log(B))) + (k*((theta_1 + theta_0)/2))  
        rejection_number = ((((sigma)**2) / (theta_1 - theta_0))*(np.log(A))) + (k*((theta_1 + theta_0)/2))
        
        #    print(p_0)
        #    print(p_1)
        if sum_up <= acceptance_number:
            print("Let us accept the Null Hypothesis H_0")
            print("The size of the population needed to end the test is: " + str(j))
            i=1000
        elif sum_up >= rejection_number: 
            print("Let us accept the Alternative Hypothesis H_1")
            print("The size of the population needed to end the test is: " + str(j))
            i=1000

        elif j == 999:
            print("End without response")
            i=1000
        else: 
            #print("repeat")
            i += 1
            j = i + 1
            k += 1
        

    def L(theta, theta_1, theta_0):
        return (((A)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0)))-1) / (((A)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0))) - ((B)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0)))) 


    try:
        theta = np.arange(-2, 2, 0.01)
        plt.plot(theta, L(theta, theta_1, theta_0), 'k')
        plt.plot(theta_0, L(theta_0, theta_1, theta_0), 'bo', label = r'$L \left( \theta_0 \right)$', color = 'b')
        plt.plot(theta_1, L(theta_1, theta_1, theta_0), 'ro', label = r'$L \left( \theta_1 \right)$', color = 'r')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$L \left( \theta \right)$')
        plt.title('Plot of the L function')
        plt.legend(loc = 'best')
        plt.show()
    except:
        print("No plot generated")
    

    def E(theta, L, sigma, theta_1, theta_0):
        return ((2*((sigma)**2)*(L*np.log(B) + (1 - L)*np.log(A)))/(2*(theta_1 - theta_0)*theta + (theta_0)**2 - (theta_1)**2))

    t = input("Enter a theta to compute the expected size of the sample: ")

    theta_fixed = float(t)

    print("The probability to end the test accepting the theta entered is: ")

    print(L(theta_fixed, theta_1, theta_0))

    print("The expected size of the sample to end the test is: ")

    print(E(theta_fixed, L(theta_fixed, theta_1, theta_0), sigma, theta_1, theta_0))


    
    print("Observations of the random variable X used to end the test: ")

    y = np.zeros(j+1)

    
    m=0
    while m <= j: 
        y[m] = s[m]
    #print(y[m])
        m += 1

    print("Graphical plotting of the observations needed: ")
    plt.plot(y)
    plt.xlabel(r'$j$')
    plt.ylabel(r'$X$')
    plt.title('Plot of the observations of X needed')
    plt.show()


    w=0
    v = np.zeros(100)
    while w < 100: 
        v[w] = theta_0
        w += 1


    r=0
    t = np.zeros(100)
    while r < 100: 
        t[r] = theta_1
        r += 1


    p=0
    k = np.zeros(100)
    while p < 100: 
        k[p] = s[p]
        p += 1
    
    print("Wider represantation of the values of X: ")


    plt.plot(k)
    plt.plot(v, 'r--', label = r'$\theta_0$', color = 'r')
    plt.plot(t, 'b--', label = r'$\theta_1$', color = 'b')
    plt.xlabel(r'$j$')
    plt.ylabel(r'$X$')
    plt.title('Plot of a sample of X')
    plt.legend(loc = 'best')
    plt.show()

    coeff = (theta_0 + theta_1)/2

    inter_0 = (((sigma)**2)/(theta_1 - theta_0))*np.log(B)

    inter_1 = (((sigma)**2)/(theta_1 - theta_0))*np.log(A)

    def L_0(coeff, inter_0, x):
        return coeff*x + inter_0

    def L_1(coeff, inter_1, x):
        return coeff*x + inter_1

    x_grid = np.arange(0, j, 0.01)

    f=0
    sum_up_plot = 0
    sum_up_plot_vector = np.zeros(j)

    while f < j:
        x[f] = s[f]
        sum_up_plot = sum_up_plot + x[f]
        sum_up_plot_vector[f] = sum_up_plot
        f += 1

    plt.plot(sum_up_plot_vector)
    plt.plot(x_grid, L_1(coeff, inter_1, x_grid), 'r', label = r'$L_1$', color = 'r')
    plt.plot(x_grid, L_0(coeff, inter_0, x_grid), 'b', label = r'$L_0$', color = 'b')
    plt.xlabel(r'$j$')
    plt.ylabel(r'$\sum_j X$')
    plt.title('Plot of the sum of X')
    plt.legend(loc = 'best')
    plt.show()

    cycle = cycle + 1



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Behaviour of the LLR function and of the observations of X")

s = np.random.normal(0, 1, 120)

sigma = 1
x = np.zeros(102)
    #print(x)
sum_up = np.zeros(102)

theta_z = input("Enter the mean before the changepoint: ") 
theta_o = input("Enter the mean after the changepoint: ") 
theta_1 = float(theta_o)
theta_0 = float(theta_z)

p_0 = 1
p_1 = 1

A = 0.95/(0.05)
B = 0.05/(0.95)

j=51
i=1

while i<51:
    x[i] = s[i]
    #print(x[i])
    p_0 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_0)**2))
    p_1 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_1)**2))
    
    sum_up[i] = sum_up[i-1] + np.log(p_1 / p_0)

    i += 1

s = np.random.normal(2, 1, 120)

while j<101:
    x[j] = s[j]
    p_0 = np.exp((-1/(2*((sigma)**2)))*((x[j] - theta_0)**2))
    p_1 = np.exp((-1/(2*((sigma)**2)))*((x[j] - theta_1)**2))
    
    sum_up[j] = sum_up[j-1] + np.log(p_1 / p_0)

    j += 1 
    
    

plt.plot(sum_up)
plt.xlabel(r'$Time$')
plt.ylabel(r'$\sum_i^n \log{f_{\theta_1} \left( X_i \right)} / {f_{\theta_0} \left( X_i \right)}$')
plt.title('Plot of the behaviour of X')
plt.show()

w=0
v = np.zeros(102)
while w < 102: 
    v[w] = theta_0
    w += 1


r=0
t = np.zeros(102)
while r < 102: 
    t[r] = theta_1
    r += 1


plt.plot(x)
plt.plot(v, 'r--', label = r'$\theta_0$', color = 'r')
plt.plot(t, 'b--', label = r'$\theta_1$', color = 'b')
plt.xlabel(r'$Time$')
plt.ylabel(r'$X$')
plt.title('Plot of the behaviour of X')
plt.legend(loc = 'best')
plt.show()














#SPRT test per una popolazione normale con due ipotesi H_0 e H_1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

number = input("How many tests do you want to run? ")
n = int(number)
print(n)
cycle = 0

while cycle < n:
    
    #print(cycle)
    print("This is a SPRT Test for a Random Normal Population (0,1) with two hypothesis H_0 and H_1")
    s = np.random.normal(0, 1, 1000)

    sigma = 1
    x = np.zeros(1000)
    #print(x)
    sum_up = 0
    theta_z = input("Enter the parameter for the Null Hypothesis H_0: ") 
    theta_o = input("Enter the parameter for the Alternative Hypothesis H_1: ") 
    theta_1 = float(theta_o)
    theta_0 = float(theta_z)
    
    alpha = 0.05
    beta = 0.05
    
    error_alpha_in = input("Enter the error of first type: ")
    error_beta_in = input("Enter the error of second type: ")
    
    error_alpha = float(error_alpha_in)
    error_beta = float(error_beta_in)
    
    A = (1 - error_beta)/error_alpha
    B = error_beta/(1 - error_alpha)
    
    j=0
    i=0
    k=1
    while i<1000:
        x[i] = s[i]
        #print(x[i])
        sum_up = sum_up + x[i]
        
        acceptance_number = ((((sigma)**2) / (theta_1 - theta_0))*(np.log(B))) + (k*((theta_1 + theta_0)/2))  
        rejection_number = ((((sigma)**2) / (theta_1 - theta_0))*(np.log(A))) + (k*((theta_1 + theta_0)/2))
        
        #    print(p_0)
        #    print(p_1)
        if sum_up <= acceptance_number:
            print("Let us accept the Null Hypothesis H_0")
            print("The size of the population needed to end the test is: " + str(j))
            i=1000
        elif sum_up >= rejection_number: 
            print("Let us accept the Alternative Hypothesis H_1")
            print("The size of the population needed to end the test is: " + str(j))
            i=1000

        elif j == 999:
            print("End without response")
            i=1000
        else: 
            #print("repeat")
            i += 1
            j = i + 1
            k += 1
        

    def L(theta, theta_1, theta_0):
        return (((A)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0)))-1) / (((A)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0))) - ((B)**((theta_1 + theta_0 - 2*theta)/(theta_1 - theta_0)))) 


    try:
        theta = np.arange(-2, 2, 0.01)
        plt.plot(theta, L(theta, theta_1, theta_0), 'k')
        plt.plot(theta_0, L(theta_0, theta_1, theta_0), 'bo', label = r'$L \left( \theta_0 \right)$', color = 'b')
        plt.plot(theta_1, L(theta_1, theta_1, theta_0), 'ro', label = r'$L \left( \theta_1 \right)$', color = 'r')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$L \left( \theta \right)$')
        plt.title('Plot of the L function')
        plt.legend(loc = 'best')
        plt.show()
    except:
        print("No plot generated")
    

    def E(theta, L, sigma, theta_1, theta_0):
        return ((2*((sigma)**2)*(L*np.log(B) + (1 - L)*np.log(A)))/(2*(theta_1 - theta_0)*theta + (theta_0)**2 - (theta_1)**2))

    t = input("Enter a theta to compute the expected size of the sample: ")

    theta_fixed = float(t)

    print("The probability to end the test accepting the theta entered is: ")

    print(L(theta_fixed, theta_1, theta_0))

    print("The expected size of the sample to end the test is: ")

    print(E(theta_fixed, L(theta_fixed, theta_1, theta_0), sigma, theta_1, theta_0))


    
    print("Observations of the random variable X used to end the test: ")

    y = np.zeros(j+1)

    
    m=0
    while m <= j: 
        y[m] = s[m]
    #print(y[m])
        m += 1

    print("Graphical plotting of the observations needed: ")
    plt.plot(y)
    plt.xlabel(r'$j$')
    plt.ylabel(r'$X$')
    plt.title('Plot of the observations of X needed')
    plt.show()


    w=0
    v = np.zeros(100)
    while w < 100: 
        v[w] = theta_0
        w += 1


    r=0
    t = np.zeros(100)
    while r < 100: 
        t[r] = theta_1
        r += 1


    p=0
    k = np.zeros(100)
    while p < 100: 
        k[p] = s[p]
        p += 1
    
    print("Wider represantation of the values of X: ")


    plt.plot(k)
    plt.plot(v, 'r--', label = r'$\theta_0$', color = 'r')
    plt.plot(t, 'b--', label = r'$\theta_1$', color = 'b')
    plt.xlabel(r'$j$')
    plt.ylabel(r'$X$')
    plt.title('Plot of a sample of X')
    plt.legend(loc = 'best')
    plt.show()

    coeff = (theta_0 + theta_1)/2

    inter_0 = (((sigma)**2)/(theta_1 - theta_0))*np.log(B)

    inter_1 = (((sigma)**2)/(theta_1 - theta_0))*np.log(A)

    def L_0(coeff, inter_0, x):
        return coeff*x + inter_0

    def L_1(coeff, inter_1, x):
        return coeff*x + inter_1

    x_grid = np.arange(0, j, 0.01)

    f=0
    sum_up_plot = 0
    sum_up_plot_vector = np.zeros(j)

    while f < j:
        x[f] = s[f]
        sum_up_plot = sum_up_plot + x[f]
        sum_up_plot_vector[f] = sum_up_plot
        f += 1

    plt.plot(sum_up_plot_vector)
    plt.plot(x_grid, L_1(coeff, inter_1, x_grid), 'r', label = r'$L_1$', color = 'r')
    plt.plot(x_grid, L_0(coeff, inter_0, x_grid), 'b', label = r'$L_0$', color = 'b')
    plt.xlabel(r'$j$')
    plt.ylabel(r'$\sum_j X$')
    plt.title('Plot of the sum of X')
    plt.legend(loc = 'best')
    plt.show()

    cycle = cycle + 1
