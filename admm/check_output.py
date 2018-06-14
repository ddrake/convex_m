from matplotlib import pyplot as plt
from mat_util import *

datadir = 'from_coeus/data'
nxs = load('nxs', datadir)
fs = load('fs', datadir)

plt.plot(fs)
plt.xlabel("Steps")
plt.ylabel("Objective Function Values")
plt.title("Convergence of Objective Function")
plt.show()

input("press a key")

plt.plot(nxs)
plt.xlabel("Steps")
plt.ylabel("1-Norm of x Values")
plt.title("Convergence of x in 1-Norm")
plt.show()


