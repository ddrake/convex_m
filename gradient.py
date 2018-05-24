from numpy import *
from matplotlib import pyplot as plt
import scipy.linalg as la

def f1(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2 + x[0] - x[1]

def f(x):
    a = array([[1,1/2],[1/2,1]])
    b = array([1,-1])
    return x.dot(a.dot(x)) + b.dot(x)

def grad_f1(x):
    gf = array([2*x[0]+x[1]+1,x[0]+2*x[1]-1])
    return gf

def grad_f(x):
    a = array([[1,1/2],[1/2,1]])
    b = array([1,-1])
    return 2*a.dot(x) + b

def gradient(max_gradf=1.0e-4, x0=[5,10], t=0.1):
    fs = []
    xk = array(x0)
    gfk = grad_f(xk)
    gfk_n2 = la.norm(gfk)
    while gfk_n2 > max_gradf:
        gfk = grad_f(xk)
        gfk_n2 = la.norm(gfk)
        xk -= t*gfk
        fk = f(xk)
        fs.append(fk)
    return array(fs), xk

def nesterov(max_gradf=1.0e-4, x0=[5,10], t=0.1):
    fs = []
    xk = array(x0)
    yk = xk
    gfk = grad_f(xk)
    gfk_n2 = la.norm(gfk)
    tk = 1
    k = 1
    while gfk_n2 > max_gradf:
        gfk = grad_f(yk)
        xk1 = yk - t*gfk
        tk1 = (1.0 + sqrt(1.0 + 4.0*tk*tk))/2.0
        g = (tk-1)/tk1
        yk = xk1 + g*(xk1 - xk)
        fk = f(xk)
        fs.append(fk)
        tk = tk1
        xk = xk1
        gfk_n2 = la.norm(gfk)
        k+=1
    return array(fs), xk

def conv_rate(alg):
    fs, x = alg()
    rs = (fs[1:]+1)/(fs[:-1]+1)
    plt.plot(rs)
    plt.show()
    return rs

