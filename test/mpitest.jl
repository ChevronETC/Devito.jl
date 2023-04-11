using MPI
MPI.Init()
myrank = MPI.Comm_rank(MPI.COMM_WORLD)
@show myrank