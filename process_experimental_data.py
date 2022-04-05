#This Python file reads and processes the data of the curing experiment.

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

def logistic_function(x, shift):
    '''Logistic function.'''
    #The 4 is fixed,
    #because the maximum slope then becomes the value of superarray
    #which is what we want
    return 1 / (1 + np.exp(shift-x*4))

def logistic_curing3d(ini, inh, time, parameters):
    '''3D version of the logistic model with an inflection point at 0.5 of the linear function.
    '''
    shift=parameters[0]
    superarray=ini[None, None, :] * 0.444 - 0.171 * inh[None, :, None]
    for i in range(11):
        for j in range(11):
            if superarray[0][i][j]>0:
                superarray[0][i][j]=superarray[0][i][j]**0.5
    C = (time[:, None, None]) * superarray
    C = logistic_function(C, shift)-logistic_function(0,shift)
    return C

def _objective_function_(paramstuple, data, ini, inh, time):
    '''
    Error for the resin model fitting.
    '''
    parameters=list(paramstuple)
    model = logistic_curing3d(ini, inh, time, parameters)
    # Error:
    diff = np.sqrt(((model - data)**2).mean())
    return diff

def fit_logistic(data, ini, inh, time):
    '''
    Fit logistic function to data, given 3 axes of ini, inh and time values.
    '''
    paramstuple=(0)
    res = minimize(_objective_function_, paramstuple, args = (data, ini, inh, time))
    return res.x

#The following function makes sure that each curve is increasing.
#Ivan thinks this is a bad idea.
#Therefore we won' use it.
def make_increasing():
 for ini in range(11):
    for inh in range(11):
        maxvalue=0
        for time in range(120):
            maxvalue=max(maxvalue,data[time][inh][ini])
            data[time][inh][ini]=maxvalue
 return
#make_increasing()


#This function reads the data for input values ini and inhi,
#And stores them inside a list.
def read(ini,inhi):
    datalist=[]
    for t in range(120):
        datalist.append(data[t][inhi][ini])
    return datalist
    
#This function plots the data for input values of ini and inhi
def plot_data(ini,inhi):
    datalist=[]
    for t in range(120):
        datalist.append(data[t][inhi][ini])
    plt.xlabel("time(seconds)")
    plt.ylabel("conversion(dimensionless)")
    times=np.linspace(0,2,num=120)
    plt.plot(times,datalist)
    return np.array(datalist)

#This function plots input values
def plot_model(ini,inhi):
    datalist=[]
    for t in range(120):
        datalist.append(curing[t][inhi][ini])
    plt.xlabel("time(seconds)")
    plt.ylabel("conversion(dimensionless)")
    times=np.linspace(0,2,num=120)
    plt.plot(times,datalist)
    return np.array(datalist)

#Calculate the resulting conversion when t -> infty
def limit(datalist):
    return datalist[119]

#Estimate the slope in the relevant part
def slope(datalist):
    L=max(datalist)
    i=118
    while(datalist[i]>0.7*L):
        i-=1
    j=i
    while(datalist[j]>0.3*L):
        j-=1
    if j==i:
        return 0 #so that we don' get an error
    return (datalist[i]-datalist[j])/((i-j)*2/119)

maxerror=0
cumulativeerror=0
limits=[]
slopes=[]
maxima=[]
maxindices=(0,0)
def processthedata():
    global maxerror, cumulativeerror,limits,slops,maxima,maxindices
    for ini in range(11):
        limits.append([])
        slopes.append([])
        maxima.append([])
        for inhi in range(11):
            modellist=[]
            datalist=[]
            for t in range(120):
                modellist.append(curing[t][inhi][ini])
                datalist.append(data[t][inhi][ini])
            limits[ini].append(round(limit(datalist),2))
            slopes[ini].append(round(slope(datalist),2))
            maxima[ini].append(round(max(datalist),2))
            modellist=np.array(modellist)
            datalist=np.array(datalist)
            diff = np.sqrt(((modellist - datalist)**2).mean())
            cumulativeerror+=diff
            if diff>maxerror:
                maxerror=diff
                maxindices=(ini,inhi)
    return

def plotslopesandmaxima():
    global slopes
    global maxima
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10,11)

    plt.figure()
    #Maximal slopes = maximal conversion rates:
    z= slopes# [[0, 0, 0, 0, 0, 0, 0.02, 0.12, 0, 0, 0], [0.47, 0.38, 0.17, 0.02, 0.34, 0.17, 0.05, 0, 0, 0, 0], [0.92, 0.81, 0.62, 0.44, 0.36, 0.07, 0.0, 0, 0, 0, 0], [1.27, 1.11, 1.06, 0.85, 0.79, 0.73, 0.59, 0.37, 0.04, 0, 0], [1.3, 1.25, 1.28, 1.1, 1.05, 1.06, 0.79, 0.82, 0.7, 0.52, 0.28], [1.42, 1.27, 1.31, 1.27, 1.32, 1.18, 1.12, 0.95, 0.8, 0.85, 0.74], [1.74, 1.53, 1.49, 1.53, 1.46, 1.47, 1.36, 1.18, 1.27, 1.2, 1.04], [1.78, 1.7, 1.67, 1.51, 1.51, 1.55, 1.28, 1.39, 1.33, 1.28, 1.22], [1.77, 1.88, 1.79, 1.75, 1.65, 1.63, 1.45, 1.48, 1.65, 1.61, 1.49], [1.84, 2.05, 2.18, 2.13, 2.0, 1.58, 1.46, 1.7, 1.82, 1.66, 1.56], [2.03, 2.06, 2.19, 2.32, 2.01, 1.54, 1.5, 1.55, 1.68, 1.68, 1.74]]
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, z, colors='black');
    plt.imshow(z,origin='lower')
    plt.xlabel('inhibiting intensity')
    plt.ylabel('initiating intensity')
    plt.title('Maximal conversion rates')
    plt.colorbar()

    plt.figure()

    #Maximal conversion values:
    z=maxima#[[0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01], [0.62, 0.58, 0.29, 0.04, 0.01, 0.01, 0.01, 0.02, 0.01, 0.0, 0.01], [1.03, 0.96, 0.84, 0.79, 0.62, 0.21, 0.03, 0.01, 0.02, 0.0, 0.01], [1.03, 1.0, 1.03, 0.95, 0.89, 0.87, 0.77, 0.44, 0.05, 0.01, 0.01], [0.99, 1.02, 1.06, 1.02, 0.88, 0.88, 0.86, 0.87, 0.83, 0.59, 0.28], [0.96, 0.97, 1.04, 1.0, 0.91, 0.86, 0.88, 0.84, 0.84, 0.87, 0.83], [1.01, 0.95, 1.02, 1.02, 0.94, 0.88, 0.86, 0.85, 0.87, 0.87, 0.86], [1.0, 0.92, 1.02, 1.02, 0.98, 0.89, 0.87, 0.88, 0.87, 0.88, 0.86], [0.94, 0.96, 1.01, 1.0, 0.98, 0.93, 0.87, 0.87, 0.89, 0.89, 0.89], [0.97, 0.99, 1.02, 1.0, 0.97, 0.95, 0.9, 0.91, 0.93, 0.89, 0.89], [0.98, 0.94, 1.03, 1.07, 1.03, 1.02, 1.0, 0.96, 0.92, 0.91, 0.91]]
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, z, colors='black');
    plt.imshow(z,origin='lower')
    plt.xlabel('inhibiting intensity')
    plt.ylabel('initiating intensity')
    plt.title('Maximal conversion values')
    plt.colorbar()
    return    

#The size of this array is 120 times 11 times 11
#the time loops over 120 values
#Here the total time is 2 seconds
#The intensitiy of initiating light goes from 0% to 100% with steps of 10%,
#and the same for the inhibiting light.

data = np.load('Amplitude data pegda 2021-12-14 no solvent__2022-02-09__1218__NS__01.npy')

#use a fit to calculate the optimal parameters
Result=list(fit_logistic(data,np.linspace(0,10,11),np.linspace(0,10,11),np.linspace(0,2,120)))
print("Optimal parameters: [shift] =",Result)

curing=logistic_curing3d(np.linspace(0,10,11),np.linspace(0,10,11),np.linspace(0,2,120),Result)

#Plot everything in one plot:
for i in range(11):
   for j in range(11):
       plot_data(i,j)
plt.title("All data")

processthedata()

plotslopesandmaxima()

print('Mean error:',cumulativeerror/121)
print('Maximal error:',maxerror)
print('Corresponding parameters: (ini,inh) =',maxindices)
#print('final values:', limits)
#print('maximal slopes:', slopes)
#print('maximal values:',maxima)

#You can change the values of ini and inh to any integers between 0 and 10
ini=3
inh=5
plt.figure()
datalist=plot_data(ini,inh)
plt.title("Conversion for ini = {} and inh = {}".format(ini,inh))
modellist=plot_model(ini,inh)
diff = np.sqrt(((modellist - datalist)**2).mean())
#print('Mean error for ini =',ini,"and inh =",inh,":", diff)