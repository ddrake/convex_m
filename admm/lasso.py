#! /usr/bin/python3
from numpy import array, zeros, mean, eye, squeeze
from numpy.random import rand
from scipy.linalg import cho_factor, cho_solve
from mat_util import load, save

def shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

newmat = True;
# for this implementation to make sense
# we need a tall thin matrix.
if newmat:
    m = 1000
    n = 15
    A = rand(m,n)
    b = rand(m,1)
    save('A',A)
    save('b',b)
else:
    A = load('A')
    b = load('b')
    m,n = a.shape
 
p = 50
lam = 1
rho = 100  # The timestep

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
  
xs = zeros((n,p));
us = zeros((n,p));
curz = zeros(n);

# Main algorithm
iters=500;
ns = zeros((iters,1));
for i in range(iters):
    for j in range(p):
        z = curz
        u = us[:,j]
        r = rlst[j]
        b = blst[j]
        a = alst[j]
        x = cho_solve(r, squeeze(a.T.dot(b)) + rho*(z-u))
        xs[:,j]=x # insert the x vector processed

    curz = shrinkage(mean(xs,1) + mean(us,1),lam/rho)
    us = (us + xs) - curz[:,newaxis]

