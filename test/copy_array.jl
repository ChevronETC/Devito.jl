using Devito, MPI, InteractiveUtils, PyCall

MPI.Init()

configuration!("mpi", true)

n = (256,512,512)

grid = Grid(shape=n, dtype=Float32)
b = Devito.Function(;name="b", grid=grid, space_order=2)

b_data_mpi = data(b)

b_data_local = zeros(Float32, n)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    b_data_local .= reshape([1:prod(n);], n)
end
MPI.Barrier(MPI.COMM_WORLD)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @info "rank0->mpi"
end
MPI.Barrier(MPI.COMM_WORLD)

copyto!(b_data_mpi, b_data_local)
t = @elapsed copyto!(b_data_mpi, b_data_local)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    tp = (prod(n)/(1000*1000*1000)) / t
    @info "t=$t, through-put=$tp GPt/s"
end
MPI.Barrier(MPI.COMM_WORLD)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @info "mpi->rank0"
end
MPI.Barrier(MPI.COMM_WORLD)

b_data_local_copy = convert(Array, b_data_mpi)
t = @elapsed begin
    b_data_local_copy = convert(Array, b_data_mpi)
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    tp = (prod(n)/1000/1000/1000) / t
    @info "t=$t, through-put=$tp GPt/s"
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @info "fill MPI array"
end
MPI.Barrier(MPI.COMM_WORLD)

fill!(b_data_mpi, 3.14f0)
t = @elapsed begin
    fill!(b_data_mpi, 3.14f0)
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    tp = (prod(n)/1000/1000/1000) / t
    @info "t=$t, through-put=$tp GPt/s"
end

MPI.Finalize()