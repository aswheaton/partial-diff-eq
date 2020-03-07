import sys
import time

from CH_Lattice import CH_Lattice

# dynamic = str(sys.argv[1])
# mode = str(sys.argv[2])
n = int(sys.argv[1])
m = int(sys.argv[2])

tic = time.clock()

simulation = CH_Lattice(a=0.1,M=0.1,K=0.1,dx=0.1,dt=0.1,animate=True,size=(n,m))
simulation.run(max_iter=1000)

toc = time.clock()
print("\nExecuted script in {} seconds.".format(toc-tic))
