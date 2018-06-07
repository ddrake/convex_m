#! /usr/bin/python3
from mpi4py import MPI
from numpy import array, zeros, mean, eye, squeeze, newaxis
from numpy.random import rand
from scipy.linalg import cho_factor, cho_solve, norm
from mat_util import load, save, save_text
from matplotlib import pyplot as plt

# Run this script like this:
# $ mpiexec -n 51 python3 lasso_distr.py

# Distributed ADMM Lasso problem
# The input data is a normally-distributed mxn random matrix
# with m = 1000, n = 15
# We will break the matrix row-wise into N=50 m_i x n skinny sub-matrices, 
# where m_i = 20
# (Later will try to break it into N=100 m_i x n fat sub-matrices,
# where m_i = 10  -- need to use the matrix inversion lemma)

# For each iteration we have two kinds of messages, each referencing a mesage type tag
# Main process sends messages to the processors with the necessary info
#   for computing the xi's.  The only vectors passed are z and the ui for the process
# Processors do the processing, each sending a message back to the main process with
#   a computed xi value.  
# Main process receives the messages with xi vectors from the processors, storing
# these vectors in an array, then averaging them along with ui vectors to get 
# xbar and ubar and from these, the new z
# Main process then computes an array of new u vectors, one for each processor.  
# These are computed by adding xi - z to the previous ui vectors 
# for each processor

def shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

debug = False
p = 50      # the number of pieces (helper processes)
lam = 10    # the weight on 1-norm of x
rho = 100

# MPI Initialization
main_id = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if debug: print("I am {}.  MPI was initialized\n".format(rank))

iters=50
if rank == main_id:
    # not necessary for solving - used by main process 
    # to generate function values for plotting
    A = load('A')
    b = load('b')
    m,n = A.shape
    # set up data to store for later plotting
    fs = zeros(iters)
    nxs = zeros(iters)

    # these arrays are managed by the main process.  Each column is a vector 
    # corresponding to one of the processors.  This makes it easy to get averages.
    xs = zeros((n,p))
    us = zeros((n,p))
    curz = zeros(n)
else:
    datadir = "data"
    A = load('A{}'.format(rank), datadir) 
    b = load('b{}'.format(rank), datadir)
    m,n = A.shape
    r = cho_factor(A.T.dot(A) + rho*eye(n))

# Main algorithm
for i in range(iters):
    if rank == main_id: # I'm the main process
        # I will send a message to each of the helpers with the data to process.
        for j in range(p):
            data = {'zj': curz, 'uj': us[:,j]}
            comm.send(data, dest=j+1, tag=11)
            if debug: print("Main sent tag 11 to {}".format(j+1))
        for j in range(p):
            data = comm.recv(tag=12, source=j+1)
            if debug: print("Main received tag 12 from {}".format(j+1))
            xs[:,j]=data['x']
        # I have all the xs for this step now - I will compute z and u  
        xm = mean(xs,1)
        um = mean(us,1)
        curz = shrinkage(xm + um,lam/rho/p)
        us = (us + xs) - curz[:,newaxis]
        # Store results of each iteration
        nx1 = norm(xm,1)
        fs[i] = 0.5*norm(A.dot(xm) - squeeze(b),2)**2 + lam * nx1
        nxs[i] = nx1
    else: # I am one of the helpers
        j = rank-1
        data = comm.recv(tag=11, source=main_id)
        if debug: print("Rank {} received tag 11 from main".format(rank))
        zj = data['zj']
        uj = data['uj']
        xj = cho_solve(r, squeeze(A.T.dot(b)) + rho*(zj-uj))
        data = {'x': xj}
        comm.send(data, dest=main_id, tag=12)
        if debug: print("Rank {} sent tag 12 to main".format(rank))

if rank == main_id:
   save('nxs',nxs)
   save('fs',fs)


