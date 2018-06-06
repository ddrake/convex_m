#! /usr/bin/python3
from mpi4py import MPI
import numpy as np
import scipy.linalg as la
import mat_utils

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

newmat = true;
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

# split the original matrix and store cholesky-factored (Ai'*Ai + rho*I)
# matrices as well as b and u vectors for use by the p processes.
# todo: pre-process the original matrix so that each process
#   just needs to load its own data from disk
 
p = 50
lambda = 1
rho = 100  # The timestep

rlst = []
blst = []
for i = in range(p):
    Ai = A[i*m/p:(i+1)*m/p]
    rlst.append(cho_factor(Ai.T*Ai + rho*eye(n)))
    blst[i] = b(i*m/p:(i+1)*m/p);
  
# these arrays are managed by the main process.  Each column is a vector 
# corresponding to one of the processors.  This makes it easy to get averages.
xs = zeros(n,p);
us = zeros(n,p);
curz = zeros(n,1);

main_id = 0;
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
printf('\n I am #d.  MPI was initialized\n',rank);

iters=500;
ns = zeros(iters,1);
for i in range(iters):
    #-----------------------------------------------------
    # send/receive messages regarding: compute x given R, b
    #----------------------------------------------------- 
    if rank == main_id # I'm the main process
        # I will send a message to each of the processors with the data to process.
        for j in range(1,p+1)
            data = {'z': curz, 'u': us(i,:)}
            comm.send(data, dest=i, tag=11)
            print("rank 0 sent message")
        for j in range(1,p+1):
            data = comm.recv(tag=12, source=j)
            print("rank 0 received tag {}".format(tag))
            xs[j]=data['x']
        # I have all the xs for this step now - I will compute z and u  
        curz = shrinkage(mean(xs,2) + mean(us,2),lambda/rho)
        us += xs - curz
    else: # I am one of the processor guys
        received = false
        while not(received):
            print("I am #{}. I am going to receive tag {}".format(rank,11))
            data = comm.recv(tag=11, source=main_id)
            z = data['z']
            u = data['u']
            r = rlst[rank] # tuple of matrix and flag constructed by cho_factor for use by cho_solve
            b = blst[rank]
            up,flag = r
            up = triu(up)
            b = blst{rank};
            # up.T(up*b) gets us back A since up is the upper triangular cholesky matrix
            x = cho_solve(r,(up.T(up*b)+rho*(z-u)));
            data = {'x': x}
            comm.send(data, dest=i, tag=12)
        received = true

def y = shrinkage(a, k):
    return (abs(a-k)-abs(a+k))/2 + a

