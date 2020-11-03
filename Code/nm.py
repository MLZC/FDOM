import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

def RosenbrockFunction(vertexs):
    if len(vertexs.shape)==1:
        x,y=vertexs
    else:
        x=vertexs[:,0]
        y=vertexs[:,1]
    return (1-x)**2+100*(y-x**2)**2

def HimmelblauFunction(vertexs):
    if len(vertexs.shape)==1:
        x,y=vertexs
    else:
        x=vertexs[:,0]
        y=vertexs[:,1]
    return (x**2+y-11)**2+(x+y**2-7)**2

def NelderMeadMethod(func,vertexs,alpha=1,beta=2,gamma=0.5,sigma=0.5):
    '''
    func: Target Function.
    alpha: the coefficient of reﬂection.
    beta: the expansion coefficient.
    gamma: the compression coefficient.
    sigma: the shrink coefficient
    '''
    f=func(vertexs)
    n,_=vertexs.shape
    #1. Order f(x_i,y_i)
    vertexs_index = np.argsort(f)
    vertexs=vertexs[vertexs_index,:]
    f=f[vertexs_index]
    #2. Calculate m, the centroid of n best vertices
    m=np.mean(vertexs[:-1,:],axis=0)
    #3. Reflection
    xk=m+alpha*(m-vertexs[-1,:])
    fxk=func(xk)
    if fxk>=f[0] and fxk<=f[-2]:
        new_vertexs = np.vstack((vertexs[:-1, :], xk))
    #4. Expansion
    elif fxk<=f[0]:
        xp=m+beta*(xk-m)
        fxp=func(xp)
        if fxp<fxk:
            new_vertexs = np.vstack((vertexs[:-1, :], xp))
        else:
            new_vertexs = np.vstack((vertexs[:-1, :], xk))
    #5. Compression
    elif fxk>f[-2]:
        xc=m+gamma*(vertexs[-1,:]-m)
        fxc=func(xc)
        if fxc<f[-1]:
            new_vertexs = np.vstack((vertexs[:-1, :], xc))
    #6. Shrink
        else:
            new_vertexs = np.zeros_like(vertexs)
            new_vertexs[0, :] = vertexs[0, :]
            for i in range(1, n):
                new_vertexs[i, :] = sigma*(vertexs[i,:] - vertexs[0,:])
    return new_vertexs


a = np.array([-5, -4])
b = np.array([-2, -2])
c = np.array([0, 2])
vertexs=np.vstack((a,b,c))
for i in range(100):
    vertexs = NelderMeadMethod(RosenbrockFunction,vertexs)
    print(vertexs[0,:],RosenbrockFunction(vertexs[0,:]))


# a = np.array([5, -1])
# b = np.array([6, 3])
# c = np.array([-5,5])
# vertexs=np.vstack((a,b,c))
# for i in range(100):
#     vertexs = NelderMeadMethod(HimmelblauFunction,vertexs)
#     print(vertexs[0,:],HimmelblauFunction(vertexs[0,:]))