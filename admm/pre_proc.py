#! /usr/bin/python3
from numpy.random import rand
from mat_util import load, save, save_text

newmat = True;
datadir = 'data'
p = 23 # number of helper processes

# we use a tall thin matrix.
if newmat:
    m = p*500 # p helpers will each solve an mxn system each iteration
    n = 150
    A = rand(m,n)
    b = rand(m,1)
    save('A',A,datadir)
    save('b',b,datadir)
else:
    A = load('A',datadir)
    b = load('b',datadir)
    m,n = A.shape
    print("m = {}, n = {}".format(m,n))

# pre-process and store the A matrix and b vector for use by the p processes.
#   each process will just need to load its own data from disk
for i in range(p):
    rank = i+1
    l = m//p
    save("A{}".format(rank), A[i*l:(i+1)*l,:], directory=datadir)
    save("b{}".format(rank), b[i*l:(i+1)*l], directory=datadir)

