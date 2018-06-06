#! /usr/bin/python3
from numpy import array, zeros, mean, eye, squeeze, newaxis
from numpy.random import rand
from scipy.linalg import cho_factor, cho_solve, norm, solve
from mat_util import load, save, save_text
from matplotlib import pyplot as plt

def shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

newmat = False;
# for this implementation to make sense
# we need a tall thin matrix.
if newmat:
    m = 1000
    n = 15
    A = rand(m,n)
    b = rand(m,1)
    save('A',A)
    save('b',b)
    save_text('A',A)
    save_text('b',b)
else:
    A = load('A')
    b = load('b')
    m,n = A.shape
    print("m = {}, n = {}".format(m,n))

lam = 10
rho = 100  # The timestep

r=cho_factor(A.T.dot(A) + rho*eye(n))
# store cholesky-factored (A'*A + rho*I)
 
x = zeros(n)
u = zeros(n)
curz = zeros(n)
Atari = A.T.dot(A) + rho*eye(n)

# Main algorithm
iters=500
fs = zeros(iters)
nxs = zeros(iters)
for i in range(iters):
    #x = solve(Atari, squeeze(A.T.dot(b)) + rho*(curz-u))
    x = cho_solve(r, squeeze(A.T.dot(b)) + rho*(curz-u))
    curz = shrinkage(x + u,lam/rho)
    u = u + x - curz
    nx1 = norm(x,1)
    fs[i] = 0.5*norm(A.dot(x) - squeeze(b))**2 + lam * nx1
    nxs[i] = nx1
plt.plot(fs)
plt.show()

