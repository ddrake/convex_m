from numpy import *
from matplotlib import pyplot as plt
import scipy.linalg as la

def f(w):
    return w.dot(xtx.dot(w)) - 2*xty.dot(w) + yty

def grad_f(w):
    return 2*(xtx.dot(w) - xty)

def gradient(max_gradf=1.0e-5, x0=[1.,1.,1.], t=0.01):
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

def load_data():
    with open('Housing.txt','r') as f:
        contents = f.read()

    lines = contents.strip().split('\n')
    text = [l.strip().split(',') for l in lines]
    data = [[float(x) for x in row] for row in text]
    x1,x2,y = zip(*data)
    x3 = tuple(1 for x in range(len(x1)))
    xx = zip(*[x1,x2,x3])
    xx = array(list(xx))
    y = array(y)
    return xx, y

def normalize(xx,y):
    xmax = xx.max(0)
    ymax = y.max()
    return xx/xmax, y/ymax, xmax, ymax

xxo, yo = load_data()
xx, y, xmax, ymax = normalize(xxo,yo)
xtx = (xx.T).dot(xx)
yty = y.dot(y)
xty = (xx.T).dot(y)
fs, w = gradient()
w = diag(1/xmax).dot(w)*ymax
price = array([2080.,4,1]).dot(w)

