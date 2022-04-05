import numpy as np
from scipy.optimize import minimize

#Fit the slopes of the experimental data to the following function:
#R = (alpha * Ini - beta * Inh)**gamma

#If you want to set gamma to 1 (linear model):
#e=1
#If you want to set gamma to 0.5 (square root model):
e=0.5
#If you want to optimize for gamma:
#e=0


def super_curing2d(ini, inh, alpha, beta, gamma):
    base=alpha*ini[:,None]-beta*inh[ None,:]
    for i in range(11):
        for j in range(11):
            base[i][j]=max(base[i][j],0.0001)

    if e>0:
        return base**e
    else:
        return base**gamma

def _objective_function_(parameters, slopes, ini, inh):
    # Compute model:
    alpha, beta, gamma = parameters
    model = super_curing2d(ini, inh, alpha, beta, gamma)

    # Error:
    diff = np.sqrt(((model - slopes)**2).mean())

    return diff

def fit_rate(slopes, ini, inh):
    '''
    Fit the function to slopes, given 2 axes of ini and inh values.
    '''
    res = minimize(_objective_function_, (0.1, 0,e), args = (slopes, ini, inh))

    return res.x

#Maximal slopes = maximal conversion rates:
slopes =  [[0, 0, 0, 0, 0, 0, 0.02, 0.12, 0, 0, 0], [0.47, 0.38, 0.17, 0.02, 0.34, 0.17, 0.05, 0, 0, 0, 0], [0.92, 0.81, 0.62, 0.44, 0.36, 0.07, 0.0, 0, 0, 0, 0], [1.27, 1.11, 1.06, 0.85, 0.79, 0.73, 0.59, 0.37, 0.04, 0, 0], [1.3, 1.25, 1.28, 1.1, 1.05, 1.06, 0.79, 0.82, 0.7, 0.52, 0.28], [1.42, 1.27, 1.31, 1.27, 1.32, 1.18, 1.12, 0.95, 0.8, 0.85, 0.74], [1.74, 1.53, 1.49, 1.53, 1.46, 1.47, 1.36, 1.18, 1.27, 1.2, 1.04], [1.78, 1.7, 1.67, 1.51, 1.51, 1.55, 1.28, 1.39, 1.33, 1.28, 1.22], [1.77, 1.88, 1.79, 1.75, 1.65, 1.63, 1.45, 1.48, 1.65, 1.61, 1.49], [1.84, 2.05, 2.18, 2.13, 2.0, 1.58, 1.46, 1.7, 1.82, 1.66, 1.56], [2.03, 2.06, 2.19, 2.32, 2.01, 1.54, 1.5, 1.55, 1.68, 1.68, 1.74]]
Result = list(fit_rate(slopes,np.linspace(0,10,11),np.linspace(0,10,11)))
print("Optimal parameters: [alpha, beta, gamma] =",Result)

Error=_objective_function_(Result,slopes,np.linspace(0,10,11),np.linspace(0,10,11))
print("Mean squared error:",Error)