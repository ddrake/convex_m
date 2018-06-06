from mpi4py import MPI
from numpy import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

m = 42

for j in range(10):
    if rank == 0:
        for i in range(1,4):
            data = {'a': j+i, 'b': 3.14}
            comm.send(data, dest=i, tag=11)
            print("rank 0 sent message")
        fsum = 0
        received = 0
        while received < 3:
            tag = 12
            data = comm.recv(tag=12)
            print("rank 0 received tag {}".format(tag))
            print(data)
            fsum += data['result']
            print("added result from {}".format(i))
            received += 1
        print("final sum is {}".format(fsum))
    elif rank >= 1:
        print("rank {} waiting for a message".format(rank))
        data = comm.recv(source=0, tag=11)
        print("rank {} received a message".format(rank))
        s = 0
        for i in range(1000000*(2-rank)):
            s += 1.0/(i+1.)/data['a']
        outdata = {'result': m*s/data['b']}
        comm.send(outdata, dest=0, tag=12)

