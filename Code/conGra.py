import numpy as np
import math
from numpy import linalg as LA
def RosenbrockFunction(x):
    return (1-x[0])**2+100*(x[1]-x[0]**2)**2

def Derivative_Rosenbrock (x):
    dx = 2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

def Hessian_Rosenbrock(x):
    dxx = 1200*x[0]**2-400*x[1]+2
    dxy = -400*x[0]
    dyx=-400*x[0]
    dyy=200
    return np.array([dxx,dxy,dyx,dyy]).reshape(2,2)

def RosenbrockFunction_alpha(x,alpha,dfx):
    x0=x[0]-alpha*dfx[0]
    x1=x[1]-alpha*dfx[1]
    return (1-x0)**2+100*(x1-x0**2)**2

def HimmelblauFunction(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

def Derivative_Himmelblau (x):
    dx = 2*(2*x[0]*(x[0]**2+x[1]-11)+x[0]+x[1]**2-7)
    dy = 2*(x[0]**2+2*x[1]*(x[0]+x[1]**2-7)+x[1]-11)
    return np.array([dx, dy])

def Hessian_Himmelblau(x):
    dxx=12*x[0]**2+4*x[1]-42
    dxy=4*(x[0]+x[1])
    dyx=4*(x[0]+x[1])
    dyy=4*x[0]+12*x[1]**2-26
    return np.array([dxx,dxy,dyx,dyy]).reshape(2,2)

def HimmelblauFunction_alpha(x,alpha,dfx):
    x0=x[0]-alpha*dfx[0]
    x1=x[1]-alpha*dfx[1]
    return (x0**2+x1-11)**2+(x0+x1**2-7)**2

# 1-dimension search algorithm
def GoldenSectionSearchMethod(x,fx,dfx,lp,rp,epsilon):
    
    '''
     Parameters:
     fx: target  function
     lp: left point
     rp: right point
     epsilon: precision
     return: minimum_point, minimum_value
    '''
    G = np.zeros((1000, 4))
    delta = rp - lp
    t = (math.sqrt(5)-1)/2
    x1 = lp + (1-t)*delta
    x2 = lp + t*delta
    fxx1 = fx(x,x1,dfx)
    fxx2 = fx(x,x2,dfx)
    k = 0
    G[k] = [lp,x1,x2,rp]
    while abs(rp-lp) > epsilon: 
        #print("iteration",k)
       # print(G[k])
        if fxx1 < fxx2:
            rp=x2
            x2 = x1
            fxx2 = fxx1
            x1 = lp + (1-t)*(rp-lp)
            fxx1 = fx(x,x1,dfx)
        else:
            lp = x1
            x1 = x2
            fxx1 = fxx2
            x2=lp+t*(rp-lp)
            fxx2=fx(x,x2,dfx)
        k = k + 1
        G[k] = lp, x1, x2, rp
    min_point = (x1+x2)/2
    min_value = fx(x,min_point,dfx)
    return min_value, min_point,G[k]

def ConjugateGradientMethod(func,dfunc,Opti_func):
    '''
    func: target function
    Opti_func: Optimization function of Alpha
    '''

    # initialize a point
    x = np.array([-1,1])
    dfx_next=dfunc(x)
    # set number of epochs
    epoch = 1000
    x_prev=None
    dfx=None
    alpha=1
    for i in range(epoch):
#         #Wolfe condition line search
#         alpha = find_step_length(func, dfunc, x, 1, -dfx_next, c2=0.1)
        # Golden Search
        _,alpha,_=GoldenSectionSearchMethod(x,Opti_func,dfx_next,1e-3,1,1e-5)
#         # Backtracking line search 
#         while func(x + alpha*(-dfunc(x))) > func(x) + c*alpha*np.dot(dfunc(x).T, (-dfunc(x))):
#             print("alpha = %.4f | f(x + alpha*(-dfunc(x))) = %.4f" %
#                   (alpha, func(x + alpha*(-dfunc(x)))))
#             alpha = rho * alpha
        print("Learning Rate:",alpha)
        # set the new point
        x_prev=x
        x=x-np.dot(alpha,dfx_next)
        # update gradient
        dfx=dfx_next
        dfx_next=dfunc(x)
        if LA.norm(x-x_prev)<=1e-5 or LA.norm(x-x_prev)<=1e-5 or LA.norm(dfx_next)<=1e-5:
            break
        # Polak-Reiber method
        Bk=(np.dot(dfx_next,dfx_next))/(np.dot(dfx,dfx))
        dfx_next=dfx_next+Bk*dfx_next

        print("epoch:",i)
        print("Min Point:"+str(x)+"\t Min Value:"+str(func(x)))

# ConjugateGradientMethod(RosenbrockFunction,Derivative_Rosenbrock,RosenbrockFunction_alpha)

ConjugateGradientMethod(HimmelblauFunction,Derivative_Himmelblau,HimmelblauFunction_alpha)