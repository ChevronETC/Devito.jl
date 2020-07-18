module Devito

using MPI, PyCall, Strided

const numpy = PyNULL()
const devito = PyNULL()
const seismic = PyNULL()

function __init__()
    copy!(numpy, pyimport("numpy"))
    copy!(devito, pyimport("devito"))
    copy!(seismic, pyimport("examples.seismic"))
end

numpy_eltype(dtype) = dtype == numpy.float32 ? Float32 : Float64

PyCall.PyObject(::Type{Float32}) = numpy.float32
PyCall.PyObject(::Type{Float64}) = numpy.float64

# Devito configuration methods
function configuration!(key, value)
    c = PyDict(devito."configuration")
    c[key] = value
    c[key]
end
configuration(key) = PyDict(devito."configuration")[key]
configuration() = PyDict(devito."configuration")

struct DevitoArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    o::PyObject # Python object for the numpy array
    p::A # copy-free
end

function DevitoArray{T,N}(o) where {T,N}
    p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), reverse(o.shape); own=false)
    DevitoArray{T,N,Array{T,N}}(o, p)
end

function DevitoArray(o)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    DevitoArray{T,N}(o)
end

Base.size(x::DevitoArray{T,N}) where {T,N} = size(x.p)
Base.parent(x::DevitoArray) = x.p

Base.getindex(x::DevitoArray{T,N,A}, i) where {T,N,A<:Array} = getindex(parent(x), i)
Base.getindex(x::DevitoArray{T,N,A}, I::Vararg{Int,N}) where {T,N,A<:StridedView} = getindex(parent(x), I...)
Base.setindex!(x::DevitoArray{T,N,A}, v, i) where {T,N,A<:Array} = setindex!(parent(x), v, i)
Base.setindex!(x::DevitoArray{T,N,A}, v, I::Vararg{Int,N}) where {T,N,A<:StridedView} = setindex!(parent(x), v, I...)
Base.IndexStyle(::Type{<:DevitoArray{<:Any,<:Any,<:Array}}) = IndexLinear()
Base.IndexStyle(::Type{<:DevitoArray{<:Any,<:Any,<:StridedView}}) = IndexCartesian()

Base.view(x::DevitoArray{T,N,Array{T,N}}, I::Vararg{Any}) where {T,N} = DevitoArray(x.o, sview(x.p, I...))

struct DevitoMPIArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    o::PyObject
    p::A
    local_indices::NTuple{N,UnitRange{Int}}
end

function DevitoMPIArray{T,N}(o, idxs) where {T,N}
    p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), length.(idxs); own=false)
    DevitoMPIArray{T,N,Array{T,N}}(o, p, idxs)
end

function Base.size(x::DevitoMPIArray{T,N}) where {T,N}
    MPI.Initialized() || MPI.Init()
    n = ntuple(i->x.local_indices[i][end], N)
    MPI.Allreduce(n, max, MPI.COMM_WORLD)
end

Base.parent(x::DevitoMPIArray) = x.p

localsize(x::DevitoMPIArray{T,N}) where {T,N} = ntuple(i->x.local_indices[i][end]-x.local_indices[i][1]+1, N)

localindices(x::DevitoMPIArray{T,N}) where {T,N} = x.local_indices

function Base.convert(::Type{Array}, x::DevitoMPIArray{T}) where {T}
    MPI.Initialized() || MPI.Init()
    y = zeros(T, size(x))
    y[localindices(x)...] .= parent(x)
    MPI.Reduce!(y, +, 0, MPI.COMM_WORLD)
    y
end

function Base.copy!(dst::DevitoMPIArray, src::AbstractArray)
    MPI.Initialized() || MPI.Init()
    MPI.Bcast!(src, 0, MPI.COMM_WORLD)
    parent(dst) .= src[localindices(dst)...]
    MPI.Barrier(MPI.COMM_WORLD)
    dst
end

function Base.fill!(x::DevitoMPIArray, v)
    MPI.Initialized() || MPI.Init()
    parent(x) .= v
    MPI.Barrier(MPI.COMM_WORLD)
    x
end

function Base.getindex(x::DevitoMPIArray{T,N}, I::Vararg{Int,N}) where {T,N}
    if all(ntuple(idim->I[idim] ∈ x.local_indices[idim], N))
        J = ntuple(idim->I[idim]-x.local_indices[idim]+1, N)
        v = getindex(x.p, J...)
    end
    v
end
Base.setindex!(x::DevitoMPIArray{T,N}, v, i) where {T,N} = @warn "not implemented"
Base.IndexStyle(::Type{<:DevitoMPIArray}) = IndexCartesian()

# TODO -- need to implement broadcasting interface for DevitoMPIArray

struct DevitoMPISparseArray{T,N,NM1} <: AbstractArray{T,N}
    o::PyObject
    p::Array
    local_indices::Array{Int,NM1}
end

function DevitoMPISparseArray{T,N}(o, idxs) where {T,N}
    local p
    if length(idxs) == 0
        p = Array{T,N}(undef, ntuple(_->0, N))
    else
        p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), reverse(o.shape); own=false)
    end
    DevitoMPISparseArray{T,N,N-1}(o, p, idxs)
end
Base.IndexStyle(::Type{<:DevitoMPISparseArray}) = IndexCartesian()
Base.getindex(x::DevitoMPISparseArray{T,N}, I::Vararg{Int,N}) where {T,N} = error("not implemented")
Base.setindex!(x::DevitoMPISparseArray{T,N}, v, I::Vararg{Int,N}) where {T,N} = error("not implemented")

Base.parent(x::DevitoMPISparseArray) = x.p

function Base.size(x::DevitoMPISparseArray{T,N}) where {T,N}
    MPI.Initialized() || MPI.Init()
    n = MPI.Allreduce(x.o.shape[2], +, MPI.COMM_WORLD)
    (n,x.o.shape[1])
end

function Base.convert(::Type{Array}, x::DevitoMPISparseArray{T,N}) where {T,N}
    MPI.Initialized() || MPI.Init()
    y = zeros(T, size(x))
    _x = parent(x)
    n = x.o.shape[1]
    for j = 1:n
        for (i,idx) in enumerate(x.local_indices)
            y[idx,j] = _x[i,j]
        end
    end
    MPI.Reduce!(y, +, 0, MPI.COMM_WORLD)
    y
end

function Base.copy!(dst::DevitoMPISparseArray, src::Array)
    MPI.Initialized() || MPI.Init()
    MPI.Bcast!(src, 0, MPI.COMM_WORLD)
    n = size(src, 2)
    dst_parent = parent(dst)
    for (i,idx) in enumerate(dst.local_indices)
        for j = 1:n
            dst_parent[i,j] = src[idx,j]
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
    dst
end

# Python <-> Julia quick-and-dirty type/struct mappings
for (M,F) in ((:devito,:Constant), (:devito,:Eq), (:devito,:Injection), (:devito,:Operator), (:devito,:SpaceDimension), (:devito,:SteppingDimension), (:devito,:TimeDimension))
    @eval begin
        struct $F
            o::PyObject
        end
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x)
        $F(args...; kwargs...) = pycall($M.$F, $F, args...; kwargs...)
        export $F
    end
end

#
# Grid
#
struct Grid{T,N}
    o::PyObject
end

function Grid(args...; kwargs...)
    o = pycall(devito.Grid, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    Grid{T,N}(o)
end

PyCall.PyObject(x::Grid) = x.o

Base.size(grid::Grid{T,N}) where {T,N} = reverse((grid.o.shape)::NTuple{N,Int})
extent(grid::Grid{T,N}) where {T,N} = reverse((grid.o.extent)::NTuple{N,T})
size_with_halo(grid::Grid{T,N}, h) where {T,N} = ntuple(i->grid.o.shape[N-i+1] + h[i][1] + h[i][2], N)
Base.size(grid::Grid, i) = size(grid)[i]
Base.ndims(grid::Grid{T,N}) where {T,N} = N
Base.eltype(grid::Grid{T}) where {T} = T

spacing(x::Union{SpaceDimension,SteppingDimension}) = x.o.spacing
spacing(x::Grid{T,N}) where {T,N} = reverse(x.o.spacing)
spacing_map(x::Grid) = PyDict(x.o."spacing_map")

#
# Functions
#
abstract type DevitoMPI end
struct DevitoMPITrue <: DevitoMPI end
struct DevitoMPIFalse <: DevitoMPI end

abstract type DiscreteFunction{T,N,M} end

struct Function{T,N,M} <: DiscreteFunction{T,N,M}
    o::PyObject
end

ismpi_distributed(o::PyObject) = o._distributor.nprocs == 1 ? DevitoMPIFalse : DevitoMPITrue  # TODO - when should should o._distributed == None ??

function Function(args...; kwargs...)
    o = pycall(devito.Function, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    M = ismpi_distributed(o)
    Function{T,N,M}(o)
end

struct TimeFunction{T,N,M} <: DiscreteFunction{T,N,M}
    o::PyObject
end

function TimeFunction(args...; kwargs...)
    o = pycall(devito.TimeFunction, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    M = ismpi_distributed(o)
    TimeFunction{T,N,M}(o)
end

struct SparseTimeFunction{T,N,M} <: DiscreteFunction{T,N,M}
    o::PyObject
end

function SparseTimeFunction(args...; kwargs...)
    o = pycall(devito.SparseTimeFunction, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    M = ismpi_distributed(o)
    SparseTimeFunction{T,N,M}(o)
end

PyCall.PyObject(x::DiscreteFunction) = x.o

grid(x::Function{T,N}) where {T,N} = Grid{T,N}(x.o.grid)
grid(x::TimeFunction{T,N}) where {T,N} = Grid{T,N-1}(x.o.grid)
halo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o.halo)::NTuple{N,Tuple{Int,Int}}
inhalo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o._size_inhalo)::NTuple{N,Tuple{Int,Int}}
Base.size(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o.shape)::NTuple{N,Int}
Base.ndims(x::DiscreteFunction{T,N}) where {T,N} = N
size_with_halo(x::DiscreteFunction{T,N}) where{T,N} = reverse(convert.(Int, x.o.shape_with_halo))::NTuple{N,Int}
size_with_inhalo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o._shape_with_inhalo)::NTuple{N,Int}

function Base.size(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N}
    MPI.Initialized() || MPI.Init()

    _shape = zeros(Int, N)
    if x.o._decomposition[1] == nothing
        if reduce(*, x.o.shape) != 0
            map(i->_shape[i] = x.o.shape[N-i+1], 1:N)
            _shape[1] = x.o.shape[N]
        end
        MPI.Reduce!(_shape, +, 0, MPI.COMM_WORLD)
    else
        error("not implemented")
    end
    ntuple(i->_shape[i], N)
end

size_with_halo(x::SparseTimeFunction) = size(x)

localmask(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_domain[N-i+1].start)+1:convert(Int,x.o._mask_domain[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}
localmask_with_halo(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_outhalo[N-i+1].start)+1:convert(Int,x.o._mask_outhalo[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}
localmask_with_inhalo(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_inhalo[N-i+1].start)+1:convert(Int,x.o._mask_inhalo[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}

localindices(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask(x)
localindices_with_halo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask_with_halo(x)
localindices_with_inhalo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask_with_inhalo(x)

forward(x::TimeFunction) = x.o.forward
backward(x::TimeFunction) = x.o.backward

data(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(DevitoArray{T,N}(x.o."_data_allocated"), localindices(x)...)
data_with_halo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(DevitoArray{T,N}(x.o."_data_allocated"), localindices_with_halo(x)...)
data_with_inhalo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(DevitoArray{T,N}(x.o."_data_allocated"), localindices_with_inhalo(x)...)
data_allocated(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = DevitoArray{T,N}(x.o."_data_allocated")

function localindices(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    ntuple(i->convert(Int,x.o.local_indices[N-i+1].start)+1:convert(Int,x.o.local_indices[N-i+1].stop), N)
end

topology(x::DiscreteFunction) = reverse(x.o._distributor.topology)
mycoords(x::DiscreteFunction) = reverse(x.o._distributor.mycoords) .+ 1
decomposition(x::DiscreteFunction) = reverse(x.o._decomposition)

function localindices_with_inhalo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    h = inhalo(x)
    localidxs = localindices(x)
    n = size_with_inhalo(x)

    _mycoords = mycoords(x)
    _decomposition = decomposition(x)

    ntuple(idim->begin
            local strt,stop
            if _decomposition[idim] == nothing
                strt = 1
                stop = n[idim]
            else
                strt = localidxs[idim][1] + (_mycoords[idim]-1)*(h[idim][1] + h[idim][2])
                stop = strt + length(localidxs[idim]) - 1 + h[idim][1] + h[idim][2]
            end
            strt:stop
        end, N)
end

function localindices_with_halo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    h = halo(x)
    localidxs = localindices(x)
    n = size_with_halo(x)

    _mycoords = mycoords(x)
    _topology = topology(x)
    _decomposition = decomposition(x)

    MPI.Initialized() || MPI.Init()
    for irnk = 0:1
        if irnk == MPI.Comm_rank(MPI.COMM_WORLD)
            @info "rnk=$irnk"
            @info "localidxs=$localidxs"
            @info "x.o.local_indices=$(x.o.local_indices)"
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end

    ntuple(idim->begin
            local strt,stop
            if _decomposition[idim] == nothing
                strt = 1
                stop = n[idim]
            else
                strt = _mycoords[idim] == 1 ? localidxs[idim][1] : localidxs[idim][1] + h[idim][1]
                stop = _mycoords[idim] == _topology[idim] ? localidxs[idim][end] + h[idim][1] + h[idim][2] : localidxs[idim][end] + h[idim][1]
            end
            strt:stop
        end, N)
end

function data(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask(x)...)
    DevitoMPIArray{T,N,typeof(p)}(x.o."_data_allocated", p, localindices(x))
end

function data_with_halo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_halo(x)...)
    DevitoMPIArray{T,N,typeof(p)}(x.o."_data_allocated", p, localindices_with_halo(x))
end

function data_with_inhalo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_inhalo(x)...)
    DevitoMPIArray{T,N,typeof(p)}(x.o."_data_allocated", p, localindices_with_inhalo(x))
end

function data_allocated(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    DevitoMPIArray{T,N}(x.o."_data_allocated", localindices_with_inhalo(x))
end

# TODO - needed? <--
function data_with_inhalo(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N}
    MPI.Initialized() || MPI.Init()
    rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    x.o._decomposition[1] == nothing || error("Sam does not know what he is doing!")
    idxs = x.o._decomposition[2][rnk+1] .+ 1
    d = DevitoMPISparseArray{T,N}(x.o."_data_allocated", idxs)
    MPI.Barrier(MPI.COMM_WORLD)
    d
end

data_with_halo(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N} = data_with_inhalo(x)
data(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N} = data_with_inhalo(x)
# -->

coordinates(x::SparseTimeFunction{T,N,DevitoMPIFalse}) where {T,N} = DevitoArray{T,N}(x.o.coordinates."_data_allocated")

function Dimension(o)
    if o.is_Space
        return SpaceDimension(o)
    elseif o.is_Stepping
        return SteppingDimension(o)
    else
        error("not implemented")
    end
end

function dimensions(x::Union{Grid{T,N},DiscreteFunction{T,N}}) where {T,N}
    ntuple(i->Dimension(x.o.dimensions[N-i+1]), N)
end

inject(x::SparseTimeFunction, args...; kwargs...) = pycall(PyObject(x).inject, Injection, args...; kwargs...)

interpolate(x::SparseTimeFunction; kwargs...) = pycall(PyObject(x).interpolate, PyObject; kwargs...)

function apply(x::Operator, args...; kwargs...)
    _summary = pycall(PyObject(x).apply, PyObject, args...; kwargs...)

    summary = Dict()
    for (k,v) in _summary.items()
        summary[k] = Dict(
            "time"=>v[1],
            "gflopss"=>v[2],
            "gpointss"=>v[3],
            "oi"=>v[4],
            "ops"=>v[5],
            "itershape"=>v[6])
    end

    summary["globals"] = Dict()
    if haskey(_summary.globals, "fdlike")
        summary["globals"]["fdlike"] = Dict(
            "time"=>_summary.globals["fdlike"][1],
            "gflopss"=>_summary.globals["fdlike"][2],
            "gpointss"=>_summary.globals["fdlike"][3],
            "oi"=>_summary.globals["fdlike"][4],
            "ops"=>_summary.globals["fdlike"][5],
            "itershape"=>_summary.globals["fdlike"][6])
    end

    if haskey(_summary.globals, "vanilla")
        summary["globals"]["vanilla"] = Dict(
            "time"=>_summary.globals["vanilla"][1],
            "gflopss"=>_summary.globals["vanilla"][2],
            "gpointss"=>_summary.globals["vanilla"][3],
            "oi"=>_summary.globals["vanilla"][4],
            "ops"=>_summary.globals["vanilla"][5],
            "itershape"=>_summary.globals["vanilla"][6])
    end
    summary
end

dx(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dx, PyObject, args...; kwargs...)
dy(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dy, PyObject, args...; kwargs...)
dz(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dz, PyObject, args...; kwargs...)

Base.:*(x::Real,y::DiscreteFunction) = PyObject(x)*PyObject(y)
Base.:*(x::DiscreteFunction, y::DiscreteFunction) = PyObject(x)*PyObject(y)
Base.:*(x::DiscreteFunction, y::PyObject) = x.o*y
Base.:*(x::PyObject, y::DiscreteFunction) = x*y.o
Base.:/(x::DiscreteFunction, y::PyObject) = x.o/y
Base.:/(x::PyObject, y::DiscreteFunction) = x/y.o
Base.:^(x::Function, y) = x.o^y

export DiscreteFunction, Grid, Function, SpaceDimension, SparseTimeFunction, SteppingDimension, TimeDimension, TimeFunction, apply, backward, configuration, configuration!, coordinates, data, data_allocated, data_with_halo, data_with_inhalo, dimensions, dx, dy, dz, extent, forward, grid, inject, interpolate, localindices, localindices_with_halo, localindices_with_inhalo, localsize, size_with_halo, spacing, spacing_map, step

end
