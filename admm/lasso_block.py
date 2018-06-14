#! /usr/bin/python3
from numpy import array, zeros, mean, eye, squeeze, newaxis
from numpy.random import rand
from scipy.linalg import cho_factor, cho_solve, norm
from mat_util import load, save, save_text
from matplotlib import pyplot as plt

def shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

newmat = False;
use_cholesky = True # Helper processes perform Cholesky factorization
datadir = 'data'
# for this implementation to make sense
# we need a tall thin matrix.
if newmat:
    m = 1000
    n = 15
    A = rand(m,n)
    b = rand(m,1)
    save('A',A,datadir)
    save('b',b,datadir)
    save_text('A',A,datadir)
    save_text('b',b,datadir)
else:
    A = load('A',datadir)
    b = load('b',datadir)
    m,n = A.shape
    print("m = {}, n = {}".format(m,n))

p = 50      # the number of pieces (helper processes)
lam = 10    # the weight on 1-norm of x
rho = 100  

# split the original matrix and store cholesky-factored (Ai'*Ai + rho*I)
# matrices as well as b and u vectors for use by the p processes.
# todo: pre-process the original matrix so that each process
#   just needs to load its own data from disk
alst = []
rlst = []
blst = []
for i in range(p):
    l = m//p
    Ai = A[i*l:(i+1)*l,:]
    alst.append(Ai)
    rlst.append(cho_factor(Ai.T.dot(Ai) + rho*eye(n)))
    blst.append(b[i*l:(i+1)*l])
  
xs = zeros((n,p))
us = zeros((n,p))
curz = zeros(n)

# Main algorithm
iters=50
fs = zeros(iters)
nxs = zeros(iters)
for i in range(iters):
    for j in range(p):
        zj = curz
        uj = us[:,j]
        rj = rlst[j]
        bj = blst[j]
        aj = alst[j]
        if use_cholesky:
            r = cho_factor(A.T.dot(A) + rho*eye(n))
        else:
            r = A.T.dot(A) + rho*eye(n)
        xj = cho_solve(rj, squeeze(aj.T.dot(bj)) + rho*(zj-uj))
        xs[:,j]=xj # insert the x vector processed

    xm = mean(xs,1)
    um = mean(us,1)
    curz = shrinkage(xm + um,lam/rho/p)
    us = (us + xs) - curz[:,newaxis]
    nx1 = norm(xm,1)
    fs[i] = 0.5*norm(A.dot(xm) - squeeze(b),2)**2 + lam * norm(xm,1)
    nxs[i] = nx1

plt.plot(fs)
plt.show()
