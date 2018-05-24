from numpy import *
from matplotlib import pyplot as plt
import scipy.linalg as la

def stochastic_subgradient(m,n, maxiter=10000000, t=0.005):
    w = ones(n)
    normw = 100
    nlist = []
    clist = []
    for k in range(maxiter):
        if k % 100000 == 0:
            print(w)
            print(normw)
            cnz = count_nonzero(y*(xx.dot(w)) < 1)
            print(cnz)
            #input("press a key")
            nlist.append(normw)
            clist.append(cnz)
        i = random.randint(0,m)
        if 1 - y[i]*xx[i,:].dot(w) > 0:
            v = -y[i]*xx[i,:]
            w = w - t*v
            normw = la.norm(w)
            if normw <= 1.0e-5:
                break
    return w, nlist, clist  

def load_train_data():
    with open('spam_train.csv','r') as f:
        contents = f.read()
    lines = contents.strip().split('\n')
    text = [l.strip().split(',') for l in lines]
    data = [[float(x) for x in row] for row in text]
    data = list(zip(*data))
    data.append([1]*len(data[0]))
    xx = array(data).T
    with open('spam_train_y.csv','r') as f:
        contents = f.read()
    text = contents.strip().split()
    data = [float(x) for x in text]
    y = array(data)
    return xx, y

def load_test_data():
    with open('spam_test.csv','r') as f:
        contents = f.read()
    lines = contents.strip().split('\n')
    text = [l.strip().split(',') for l in lines]
    data = [[float(x) for x in row] for row in text]
    data = list(zip(*data))
    data.append([1]*len(data[0]))
    xx = array(data).T
    with open('spam_test_y.csv','r') as f:
        contents = f.read()
    text = contents.strip().split()
    data = [float(x) for x in text]
    y = array(data)
    return xx, y



xx, y = load_train_data()
m,n = xx.shape
w, nlist, clist = stochastic_subgradient(m,n)
xx, y = load_test_data()
ytest = xx.dot(w)
matches = count_nonzero(ytest*y > 0)
tcount = len(ytest)

