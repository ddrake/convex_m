#! /usr/bin/python3
from numpy.random import rand
from mat_util import load, save, save_text

newmat = False;
# we use a tall thin matrix.
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

p = 50

# pre-process and store the A matrix and b vector for use by the p processes.
#   each process will just need to load its own data from disk
datadir = "data"
for i in range(p):
    rank = i+1
    l = m//p
    save("A{}".format(rank), A[i*l:(i+1)*l,:], directory=datadir)
    save("b{}".format(rank), b[i*l:(i+1)*l], directory=datadir)

