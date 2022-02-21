# DEVITO_LOGGING=DEBUG DEVITO_LANGUAGE=openmp DEVITO_MPI=1 mpirun -n 30 -bind-to core:4 -map-by l3cache /bin/python mpiset.py



import numpy as np
import time
from devito.mpi import MPI
from devito import (configuration, Grid, Function)

configuration["mpi"] = True

space_order = 8
nx,ny,nz = 1201,1201,601
shape = (nx, ny, nz)
spacing = (10, 10, 10)
origin = (0, 0, 0)
extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
grid = Grid(extent=extent, shape=shape, origin=origin, dtype=np.float32)


f1 = Function(name='f1', grid=grid, space_order=space_order)


f2 = np.zeros((nx,ny,nz), dtype=np.float32)


t1a = time.time()
f2[:] = 1
t1b = time.time()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    print("")
    print("time to fill numpy array; %10.4f sec\n" % (t1b - t1a))


t2a = time.time()
f1.data[:] = f2[:]
t2b = time.time()


if rank == 0:
    print("time to copy mpi array; %10.4f sec\n" % (t2b - t2a))