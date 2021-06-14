import numpy as np
from math import *

phi = (1.0 + sqrt(5.0))/2.0

def golden_section_searcher(X, d, prev_val, lower, upper, epsilon):
    
    x1 = upper - ((phi - 1)*(upper - lower))
    x2 = lower + ((phi - 1)*(upper - lower))
    val = x1
    
    param2 = X - np.dot(x2, d)
    param2 = param2.tolist()
    
    param1 = X - np.dot(x1, d)
    param1 = param1.tolist()
    
    if equation(param2) < equation(param1):
        if x1 > x2:
            upper = x1
        else:
            lower = x1

    else:
        if x2 > x1:
            upper = x2
        else:
            lower = x2

    if abs(prev_val - val) <= epsilon:
        return val
    else:
        return golden_section_searcher(X, d, val, lower, upper, epsilon)

def derivate(f, X):
    h = 0.0000001
    delf = []
    
    for i in range(len(X)):
        E = np.zeros(len(X))
        E[i] = h
        vals = X + E
        delf.append((f(vals) - f(X))/h)
            
    return delf


def difference(X, Y):
    total = 0
    
    for i in range(len(X)):
        total = total + abs(X[i] - Y[i])
    total = total / len(X)
    

    return total


def steepest_descent(X, epsilon):
    
    while True:
        d = derivate(equation, X)
        x_prev = X

        learning_rate = golden_section_searcher(X, d, 1, -10, 10, 0.0001)
        X = X - np.dot(learning_rate, d)
        X = X.tolist()
        
        if difference(x_prev, X) < epsilon:
            return x_prev
        
        
    return x_prev

def equation(x):
    return (x[0]+5)**2 + (x[1]+8)**2 + (x[2]+7)**2 + (2*x[0]**2)*x[1]**2 + (4*x[0]**2)*x[2]**2
    
inputs = np.array([1,1,1]).transpose()   
results = steepest_descent(inputs, 0.0000001)
print(steepest_descent(inputs, 0.0000001))
