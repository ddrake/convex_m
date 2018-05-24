from numpy import *
import scipy.linalg as la

from matplotlib import pyplot as plt

h = la.hilbert(5)
def f(x):
    return x.dot(la.hilbert(5).dot(x))

def grad_f(x):
    return 2*la.hilbert(5).dot(x)

def gradient(max_gradf=1.0e-2, x0=[1.,2.,3.,4.,5.], t=0.1):
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

def conv_rate(alg):
    fs, x = alg()
    rs = (fs[1:]+1)/(fs[:-1]+1)
    plt.plot(rs)
    plt.show()
    return rs

