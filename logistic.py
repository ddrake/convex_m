from numpy import *
from matplotlib import pyplot as plt
import scipy.linalg as la

def f(w):
    return sum(log(1+exp(xx.dot(w))) - y*xx.dot(w))

def sigma(t):
    return 1./(1+exp(-t))

def grad_f(w):
    sigmy = sigma(xx.dot(w)) - y
    sdx = sigmy.dot(xx)
    return sdx

# if you use the more correlated data set Logistic1,
# need to change the t value to 0.1 or convergence is slow
def gradient(max_gradf=1.0e-4, x0=[1.,1.], t=0.01):
    fs = []
    xk = array(x0)
    gfk = grad_f(xk)
    gfk_n2 = la.norm(gfk)
    while gfk_n2 > max_gradf:
        gfk = grad_f(xk)
        #print(gfk)
        gfk_n2 = la.norm(gfk)
        xk -= t*gfk
        fk = f(xk)
        fs.append(fk)
    return array(fs), xk

def nesterov(max_gradf=1.0e-4, x0=[1.,1.], t=0.01):
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

def load_data():
    with open('LogisticData.txt','r') as f:
        contents = f.read()

    lines = contents.strip().split('\n')
    text = [l.strip().split() for l in lines]
    data = [[float(x) for x in row] for row in text]
    x,y = data
    xx = list(zip(*[x,[1]*len(x)]))
    xx, y = array(xx), array(y)
    return xx, y, data

xx, y, data = load_data()
fs, w = nesterov()
