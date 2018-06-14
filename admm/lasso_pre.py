#! /usr/bin/python3
# Distributed ADMM Lasso problem
# The input data A and b where A is a normally-distributed mxn random matrix
# with m = 1000, n = 15 and b is a normally-distributed mx1 random vector
# 
# Before running this, we use 'pre_proc.py' to preprocess A and b, 
# breaking each row-wise into N=50 m_i x n skinny sub-matrices, 
# where m_i = 20

# Run this script like this:
# $ mpiexec -n 51 python3 lasso_pre.py

from mpi4py import MPI
from numpy import array, zeros, empty, mean, eye, squeeze, newaxis
from scipy.linalg import cho_factor, cho_solve, solve, norm
from mat_util import load, save, save_text
from datetime import datetime

# shrinkage or prox function (fastest python implementation)
def shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

use_cholesky = True # Helper processes perform Cholesky factorization
debug = False       # Print debug information for MPI
local = True

# data directory for coeus cluster
if local:
    datadir = 'data'
else:
    datadir = '/scratch/ddrake/admm/data'

p   = 23            # the number of pieces (helper processes)
lam = 10            # the weight on 1-norm of x
rho = 100

# MPI Initialization
main_id = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if debug: print("I am {}.  MPI was initialized\n".format(rank))

iters=50
if rank == main_id:
    print("started at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # A and b are not necessary for solving.
    # The are used here by the main process 
    # to generate function values for plotting
    A = load('A',datadir)
    b = load('b',datadir)
    m,n = A.shape

    # set up data to store for later plotting
    fs = zeros(iters)
    nxs = zeros(iters)

    # these arrays are managed by the main process.  Each column is a vector 
    # corresponding to one of the processors.  
    # This makes it easy to get averages.
    xs = zeros((n,p))
    us = zeros((n,p))
    z = zeros(n)
else:
    Aj = load('A{}'.format(rank), datadir) 
    bj = load('b{}'.format(rank), datadir)
    m,n = Aj.shape
    if use_cholesky:
        r = cho_factor(Aj.T.dot(Aj) + rho*eye(n))
        print("Rank {} finished factoring at {}" \
                .format(rank, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    else:
        r = Aj.T.dot(Aj) + rho*eye(n)

# Main algorithm
for i in range(iters):
    if rank == main_id: # I'm the main process
        # I'll send a message to each of the helpers with u and v.
        for j in range(p):
            data = {'zj': z, 'uj': us[:,j]}
            comm.send(data, dest=j+1, tag=11)
            if debug: print("Main sent tag 11 to {}".format(j+1))
        for j in range(p):
            data = comm.recv(tag=12, source=j+1)
            if debug: print("Main received tag 12 from {}".format(j+1))
            xs[:,j]=data['x']
        # I have all the xs for this step now - I'll compute z and u  
        xm = mean(xs,1)
        um = mean(us,1)
        z = shrinkage(xm + um,lam/rho/p)
        us = (us + xs) - z[:,newaxis]
        # Storing the results of this iteration for plotting
        nx1 = norm(xm,1)
        fs[i] = 0.5*norm(A.dot(xm) - squeeze(b),2)**2 + lam * nx1
        nxs[i] = nx1
    else: # I am one of the helpers
        j = rank-1
        data = comm.recv(tag=11, source=main_id)
        if debug: print("Rank {} received tag 11 from main".format(rank))
        zj = data['zj']
        uj = data['uj']
        if use_cholesky:
            xj = cho_solve(r, squeeze(Aj.T.dot(bj)) + rho*(zj-uj))
        else:
            xj = solve(r, squeeze(Aj.T.dot(bj)) + rho*(zj-uj))
        data = {'x': xj}
        comm.send(data, dest=main_id, tag=12)
        if debug: print("Rank {} sent tag 12 to main".format(rank))

if rank == main_id:
    # Save results for plotting by 'check_output.py' 
    save('nxs',nxs,datadir)
    save('fs',fs,datadir)
    print("finished at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
