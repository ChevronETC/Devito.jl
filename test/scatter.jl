using MPI

MPI.Init()

function myscatter(data_rank_zero::AbstractArray{T,N}) where {T,N}
    MPI.Initialized() || MPI.Init()

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        leadingdims = size(data_rank_zero)[1:N-1]
        n = size(data_rank_zero, N) # split the last dimension
        q,r = divrem(n, MPI.Comm_size(MPI.COMM_WORLD))
        counts = mapreduce(i->i <= r ? [leadingdims..., q+1] : [leadingdims..., q], vcat, 1:MPI.Comm_size(MPI.COMM_WORLD))
        @show counts
        # counts = [i <= r ? (leadingdims..., q+1) : (leadingdims..., q) for i=1:MPI.Comm_size(MPI.COMM_WORLD)]

        size_ubuffer = UBuffer(counts, N)
        data_vbuffer = VBuffer(data_rank_zero, counts)
    else
        size_ubuffer = UBuffer(nothing)
        data_vbuffer = VBuffer(nothing)
    end

    local_size = MPI.Scatter(size_ubuffer, Int, 0, MPI.COMM_WORLD)
    local_data = MPI.Scatterv!(data_vbuffer, zeros(Float64, local_size), 0, MPI.COMM_WORLD)

    local_data
end

function mybcast(data_rank_zero)
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    comm_rank = MPI.Comm_rank(comm)

    MPI.Bcast!(data_rank_zero, 0, comm)
    n = length(data_rank_zero)
    q,r = divrem(n, comm_size)
    counts = [i <= r ? q+1 : q for i=1:comm_size]

    i1 = comm_rank == 0 ? 1 : 1 + sum(counts[1:comm_rank])
    i2 = i1 + counts[comm_rank+1] - 1
    local_data = data_rank_zero[i1:i2]

    local_data
end

n = 256*256*256
x = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? rand(n) : zeros(n)

_xs = myscatter(x)
tscatter = @elapsed begin
    _xs = myscatter(x)
end

_xb = mybcast(x)
tbcast = @elapsed begin
    _xb = mybcast(x)
end

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @info "rank 0"
    @info "scatter: $tscatter seconds, $((length(x)/(1000*1000*1000))/tscatter) GPt/s"
    @info "broadcast: $tbcast seconds, $((length(x)/(1000*1000*1000))/tbcast) GPt/s"
    # @show x
    # @show _xs
    # @show _xs2
    # @show _xb
end
# MPI.Barrier(MPI.COMM_WORLD)

# if MPI.Comm_rank(MPI.COMM_WORLD) == 1
#     @info "rank 1"
#     @show _xs
#     @show _xb
# end
