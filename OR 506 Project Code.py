# -*- coding: utf-8 -*-
"""
    OR 506 Project
    Author: Thad M.
"""
import numpy as np

# The values for the geometric means(mu) and the covariances were calcualted in excel

# Define the list of mu values and the covariance matrix
mu = np.array([[0.068908],
               [0.080776],
               [0.026875],
               [0.030986]])

muT = np.transpose(mu)

cov = np.array([[0.028261, 0.037357, 0.013997, 0.012885],
                  [0.037357, 0.549892, -0.01593, -0.02209],
                  [0.013997, -0.01593, 0.025807, 0.028521],
                  [0.012885, -0.02209, 0.028521, 0.050686]])

# Define initial feasible starting point x0
x0 = np.array([[0.25],
               [0.25],
               [0.25],
               [0.25]])

# Define objective function f(x) and the gradient of f(x)
def f(x1, x2, x3, x4, pen):
    return (1 / 4) * np.matmul((np.matmul(np.transpose(np.array([[x1],[x2],[x3], [x4]])), cov)), np.array([[x1],[x2],[x3], [x4]])) - (1/2) * np.matmul(muT, np.array([[x1],[x2],[x3], [x4]])) + pen * ((x1 + x2 + x3 + x4 - 1 ) ** 2)

def fgrad(x1, x2, x3, x4, pen) :
    return (1/2) * np.matmul(cov, np.array([[x1],[x2],[x3], [x4]])) - (1/2) * mu + (pen * np.array([[2* (x1 + x2 + x3 + x4 - 1 )], [2* (x1 + x2 + x3 + x4 - 1 )], [2* (x1 + x2 + x3 + x4 - 1 )], [2* (x1 + x2 + x3 + x4 - 1 )]]))

# Define stopping criteria values 
epsilon = 10 ** -8
itmax = 10000

# Set initial guess to be x1 
x1 = x0

# Set initial x_k value to be x1
x_k = x1

# Setup additional parameters for iterations
k = 1
a = 10 ** -4
n = 0.9

# Setup penatly parameter 
pen = 100

# Steup display table
print("k                    ", "x_k                                       ", "f(x_k)")
print(k,"      ", np.transpose(x_k), "                        ", f(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen))

# Begin iterations and setup stopping criteria 
# This while loop perform's Cauchy's Method of Steepest Descent
while k < itmax and (np.linalg.norm(fgrad(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)) / (1 + abs(f(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)))) > epsilon: 
    k = k + 1                                               ## Move to next iteration (k + 1)
    d_k = -1 * fgrad(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)              ## Calculate new direction of descent (dk = -gradient(xk))
    lambda_k = 1                                            ## Set initial lambda_k value to be 1 (based on class notes)
    x_k1 = x_k + lambda_k * d_k                             ## Calculate new x_k value based on starting lambda_k
    ## This while loop is for Goldstein-Armijo Criteria
    ## The @ symbol performs matrix multiplication 
    while (f(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen) + a * lambda_k * np.transpose(d_k) @ (fgrad(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)) < f(x_k1[0,0], x_k1[1,0], x_k1[2,0], x_k1[3,0], pen)) or (n * np.transpose(d_k) @ fgrad(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)) > (np.transpose(d_k) @ fgrad(x_k1[0,0], x_k1[1,0], x_k1[2,0], x_k1[3,0], pen)):
        lambda_k = lambda_k * 0.5                           ## Alter lambda value based on class notes
        x_k1 = x_k + lambda_k * d_k                         ## Change x_k1 value based on new lambda_k value
    x_k = x_k1                                              ## Replace x_k value with x_k1 before next iteration
    if k <= 5 or k >= 9996:                                  ## This shrinks the results table so that it is easier to read
        print(k,"   ", np.transpose(x_k), "   ", f(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen))    ## Add iteration to results table
 
print()
## Create final results table
print("Iterations Complete, Stopping Criteria Met")
print("Minimum found at x1 =", x_k[0,0], " x2=", x_k[1,0]) 
print("x3=", x_k[2,0], " x4=", x_k[3,0])            
print("Minimum value is at f(x) =", np.asscalar(f(x_k[0,0], x_k[1,0], x_k[2,0], x_k[3,0], pen)))      


