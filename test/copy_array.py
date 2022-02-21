# DEVITO_LOGGING=DEBUG DEVITO_LANGUAGE=openmp DEVITO_MPI=1 mpirun -n 30 -bind-to core:4 -map-by l3cache /bin/python mpiset.py
import numpy as np
import time
from devito.mpi import MPI
from devito import (configuration, Grid, Function)

configuration["mpi"] = True

nx,ny,nz = 256,512,512
shape = (nx, ny, nz)

grid = Grid(shape=shape, dtype=np.float32)
b = Function(name="b", grid=grid, space_order=2)

b_data_mpi = b.data

b_data_local = np.zeros((nx,ny,nz), dtype=np.float32)
if MPI.COMM_WORLD.Get_rank() == 0:
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                b_data_local[ix,iy,iz] = ix*iy*iz
MPI.COMM_WORLD.Barrier()

if MPI.COMM_WORLD.Get_rank() == 0:
    print("rank0->mpi")
MPI.COMM_WORLD.Barrier()

t1 = time.time()
b_data_mpi[:] = b_data_local[:]
t2 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    t = t2 - t1
    throughput = ((nx*ny*nz)/(1000*1000*1000))/t
    print(f't={t}, through-put={throughput} GPt/s')
MPI.COMM_WORLD.Barrier()

if MPI.COMM_WORLD.Get_rank() == 0:
    print("mpi->rank0")
MPI.COMM_WORLD.Barrier()

t1 = time.time()
_b_data_local = b_data_mpi[:]
t2 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    t = t2 - t1
    throughput = ((nx*ny*nz)/(1000*1000*1000))/t
    print(f't={t}, through-put={throughput} GPt/s')
MPI.COMM_WORLD.Barrier()

if MPI.COMM_WORLD.Get_rank() == 0:
    print("fill MPI array")
MPI.COMM_WORLD.Barrier()

t1 = time.time()
b_data_mpi[:] = 1.0
t2 = time.time()

if MPI.COMM_WORLD.Get_rank() == 0:
    t = t2 - t1
    throughput = ((nx*ny*nz)/(1000*1000*1000))/t
    print(f't={t}, through-put={throughput} GPt/s')
MPI.COMM_WORLD.Barrier()