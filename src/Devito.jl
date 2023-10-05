module Devito

using LinearAlgebra, MPI, PyCall, Strided

const numpy = PyNULL()
const sympy = PyNULL()
const devito = PyNULL()
const devitopro = PyNULL()
const seismic = PyNULL()

has_devitopro() = get(ENV, "DEVITO_PRO", "") != ""

function __init__()
    try
        copy!(numpy, pyimport("numpy"))
        copy!(sympy, pyimport("sympy"))
        copy!(devito, pyimport("devito"))
        copy!(devitopro, (has_devitopro() ? pyimport("devitopro") : pyimport("devito")))
        copy!(seismic, pyimport("examples.seismic"))
    catch e
        if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
            @warn "unable to pyimport"
        else
            throw(e)
        end
    end   
end

numpy_eltype(dtype) = dtype == numpy.float32 ? Float32 : Float64

PyCall.PyObject(::Type{Float32}) = numpy.float32
PyCall.PyObject(::Type{Float64}) = numpy.float64

"""
    configuration!(key, value)

Configure Devito.  Examples include
```julia
configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", false)
```
"""
function configuration!(key, value)
    c = PyDict(devito."configuration")
    c[key] = value
    c[key]
end
configuration(key) = PyDict(devito."configuration")[key]
configuration() = PyDict(devito."configuration")

_reverse(argument::Tuple) = reverse(argument)
_reverse(argument) = argument

function reversedims(arguments)
    _arguments = collect(arguments)
    keys = first.(_arguments)
    values = @. _reverse(last(_arguments))    
    (; zip(keys, values)...)
 end

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

abstract type DevitoMPIAbstractArray{T,N} <: AbstractArray{T,N} end

Base.parent(x::DevitoMPIAbstractArray) = x.p
localsize(x::DevitoMPIAbstractArray{T,N}) where {T,N} = ntuple(i->size(x.local_indices[i])[1], N)
localindices(x::DevitoMPIAbstractArray{T,N}) where {T,N} = x.local_indices
decomposition(x::DevitoMPIAbstractArray) = x.decomposition
topology(x::DevitoMPIAbstractArray) = x.topology

function _size_from_local_indices(local_indices::NTuple{N,UnitRange{Int64}}) where {N}
    n = ntuple(i->(size(local_indices[i])[1] > 0 ? local_indices[i][end] : 0), N)
    MPI.Allreduce(n, max, MPI.COMM_WORLD)
end

Base.size(x::DevitoMPIAbstractArray) = x.size

function counts(x::DevitoMPIAbstractArray)
    [count(x, mycoords) for mycoords in CartesianIndices(topology(x))][:]
end

function Base.fill!(x::DevitoMPIAbstractArray, v)
    parent(x) .= v
    MPI.Barrier(MPI.COMM_WORLD)
    x
end

Base.IndexStyle(::Type{<:DevitoMPIAbstractArray}) = IndexCartesian()

struct DevitoMPIArray{T,N,A<:AbstractArray{T,N},D} <: DevitoMPIAbstractArray{T,N}
    o::PyObject
    p::A
    local_indices::NTuple{N,UnitRange{Int}}
    decomposition::D
    topology::NTuple{N,Int}
    size::NTuple{N,Int}
end

function DevitoMPIArray{T,N}(o, idxs, decomp::D, topo) where {T,N,D}
    p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), length.(idxs); own=false)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,Array{T,N},D}(o, p, idxs, decomp, topo, n)
end

function convert_resort_array!(_y::Array{T,N}, y::Vector{T}, topology, decomposition) where {T,N}
    i = 1
    for block_idx in CartesianIndices(topology)
        idxs = CartesianIndices(ntuple(idim->decomposition[idim] === nothing ? size(_y, idim) : length(decomposition[idim][block_idx.I[idim]]), N))
        for _idx in idxs
            idx = CartesianIndex(ntuple(idim->decomposition[idim] === nothing ? _idx.I[idim] : decomposition[idim][block_idx.I[idim]][_idx.I[idim]], N))
            _y[idx] = y[i]
            i += 1
        end
    end
    _y
end

function Base.convert(::Type{Array}, x::DevitoMPIAbstractArray{T,N}) where {T,N}
    local y
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        y = zeros(T, length(x))
        y_vbuffer = VBuffer(y, counts(x))
    else
        y = Array{T}(undef, ntuple(_->0, N))
        y_vbuffer = VBuffer(nothing)
    end
    
    _x = zeros(T, size(parent(x)))

    copyto!(_x, parent(x))
    MPI.Gatherv!(_x, y_vbuffer, 0, MPI.COMM_WORLD)
                        
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
    else
        _y = Array{T,N}(undef, ntuple(_->0, N))
    end
    _y
end

function copyto_resort_array!(_y::Vector{T}, y::Array{T,N}, topology, decomposition) where {T,N}
    i = 1
    for block_idx in CartesianIndices(topology)
        idxs = CartesianIndices(ntuple(idim->decomposition[idim] === nothing ? size(y, idim) : length(decomposition[idim][block_idx.I[idim]]), N))
        for _idx in idxs
            idx = CartesianIndex(ntuple(idim->decomposition[idim] === nothing ? _idx.I[idim] : decomposition[idim][block_idx.I[idim]][_idx.I[idim]], N))
            _y[i] = y[idx]
            i += 1
        end
    end
    _y
end

function Base.copyto!(dst::DevitoMPIArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    _counts = counts(dst)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
        data_vbuffer = VBuffer(_y, _counts)
    else
        data_vbuffer = VBuffer(nothing)
    end

    _dst = MPI.Scatterv!(data_vbuffer, Vector{T}(undef, _counts[MPI.Comm_rank(MPI.COMM_WORLD)+1]), 0, MPI.COMM_WORLD)
    copyto!(parent(dst), _dst)
end

struct DevitoMPITimeArray{T,N,A<:AbstractArray{T,N},NM1,D} <: DevitoMPIAbstractArray{T,N}
    o::PyObject
    p::A
    local_indices::NTuple{N,UnitRange{Int}}
    decomposition::D
    topology::NTuple{NM1,Int}
    size::NTuple{N,Int}
end

function DevitoMPITimeArray{T,N}(o, idxs, decomp::D, topo::NTuple{NM1,Int}) where {T,N,D,NM1}
    p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), length.(idxs); own=false)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,Array{T,N},NM1,D}(o, p, idxs, decomp, topo, n)
end

function Base.convert(::Type{Array}, x::DevitoMPITimeArray{T,N}) where {T,N}
    local y
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        y = zeros(T, length(x))
        y_vbuffer = VBuffer(y, counts(x))
    else
        y = Vector{T}(undef, 0)
        y_vbuffer = VBuffer(nothing)
    end
    MPI.Gatherv!(convert(Array, parent(x)), y_vbuffer, 0, MPI.COMM_WORLD)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
    else
        _y = zeros(T, ntuple(_->0, N))
    end

    _y
end

function Base.copy!(dst::DevitoMPIAbstractArray, src::AbstractArray)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        axes(dst) == axes(src) || throw(ArgumentError(
            "arrays must have the same axes for copy! (consider using `copyto!`)"))
    end
    copyto!(dst, src)
end

function Base.copy!(dst::DevitoMPIAbstractArray{T,1}, src::AbstractVector) where {T}
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        axes(dst) == axes(src) || throw(ArgumentError(
            "arrays must have the same axes for copy! (consider using `copyto!`)"))
    end
    copyto!(dst, src)
end

function Base.copyto!(dst::DevitoMPITimeArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    _counts = counts(dst)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
        data_vbuffer = VBuffer(_y, _counts)
    else
        data_vbuffer = VBuffer(nothing)
    end

    _dst = MPI.Scatterv!(data_vbuffer, Vector{T}(undef, _counts[MPI.Comm_rank(MPI.COMM_WORLD)+1]), 0, MPI.COMM_WORLD)
    copyto!(parent(dst), _dst)
end

struct DevitoMPISparseTimeArray{T,N,NM1,D} <: DevitoMPIAbstractArray{T,NM1}
    o::PyObject
    p::Array{T,NM1}
    local_indices::NTuple{NM1,Vector{Int}}
    decomposition::D
    topology::NTuple{NM1,Int}
    size::NTuple{NM1,Int}
end

function DevitoMPISparseTimeArray{T,N}(o, idxs, decomp::D, topo::NTuple{NM1,Int}) where {T,N,D,NM1}
    local p
    if length(idxs) == 0
        p = Array{T,N}(undef, ntuple(_->0, N))
    else
        p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), length.(idxs); own=false)
    end
    DevitoMPISparseTimeArray{T,N,NM1,D}(o, p, idxs, decomp, topo, globalsize(decomp))
end

localsize(x::DevitoMPISparseTimeArray) = length.(x.local_indices)


struct DevitoMPISparseArray{T,N,NM1,D} <: DevitoMPIAbstractArray{T,N}
    o::PyObject
    p::Array{T,NM1}
    local_indices::NTuple{NM1,Vector{Int}}
    decomposition::D
    topology::NTuple{NM1,Int}
    size::NTuple{NM1,Int}
end

function DevitoMPISparseArray{T,N}(o, idxs, decomp::D, topo::NTuple{NM1,Int}) where {T,N,D,NM1}
    local p
    if prod(length.(idxs)) == 0
        p = Array{T,N}(undef, ntuple(_->0, N))
    else
        p = unsafe_wrap(Array{T,N}, Ptr{T}(o.__array_interface__["data"][1]), length.(idxs); own=false)
    end
    DevitoMPISparseArray{T,N,NM1,D}(o, p, idxs, decomp, topo, globalsize(decomp))
end

localsize(x::DevitoMPISparseArray) = length.(x.local_indices)

globalsize(decomp) = ntuple( i -> max(cat(decomp[i]..., dims=1)...) - min(cat(decomp[i]..., dims=1)...) + 1 , length(decomp))

function count(x::Union{DevitoMPIArray,DevitoMPITimeArray,DevitoMPISparseArray,DevitoMPISparseTimeArray}, mycoords)
    d = decomposition(x)
    n = size(x)
    mapreduce(idim->d[idim] === nothing ? n[idim] : length(d[idim][mycoords[idim]]), *, 1:length(d))
end

function Base.convert(::Type{Array}, x::Union{DevitoMPISparseTimeArray{T,N},DevitoMPISparseArray{T,N}}) where {T,N}
    local y
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        y = zeros(T, length(x))
        y_vbuffer = VBuffer(y, counts(x))
    else
        y = Array{T,N}(undef, ntuple(_->0, N))
        y_vbuffer = VBuffer(nothing)
    end
    _x = zeros(T, size(parent(x)))
    copyto!(_x, parent(x))
    MPI.Gatherv!(_x, y_vbuffer, 0, MPI.COMM_WORLD)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
    else
        _y = Array{T,N}(undef, ntuple(_->0, N))
    end
    _y
end

function Base.copyto!(dst::Union{DevitoMPISparseTimeArray{T,N},DevitoMPISparseArray{T,N}}, src::Array{T,N}) where {T,N}
    _counts = counts(dst)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
        data_vbuffer = VBuffer(_y, _counts)
    else
        data_vbuffer = VBuffer(nothing)
    end
    _dst = MPI.Scatterv!(data_vbuffer, Vector{T}(undef, _counts[MPI.Comm_rank(MPI.COMM_WORLD)+1]), 0, MPI.COMM_WORLD)
    copyto!(parent(dst), _dst)
end

function in_range(i::Int, ranges)
    for rang in enumerate(ranges)
        if i ∈ rang[2]
            return rang[1]
        end
    end
    error("Outside Valid Ranges")
end

function helix_helper(tup::NTuple{N,Int}) where {N}
    wrapper = (1,)
    for i in 2:N
        wrapper = (wrapper..., wrapper[1]*tup[i-1])
    end
    return wrapper
end

function find_rank(x::DevitoMPIAbstractArray{T,N}, I::Vararg{Int,N}) where {T,N}
    decomp = decomposition(x)
    rank_position = in_range.(I,decomp)
    helper = helix_helper(topology(x))
    rank = sum((rank_position .- 1) .* helper)
    return rank
end

shift_localindicies(i::Int, indices::Union{UnitRange{Int},Vector{Int}}) = i - indices[1] + 1

shift_localindicies(i::Int, indices::Int) = i - indices + 1

function Base.getindex(x::DevitoMPIAbstractArray{T,N}, I::Vararg{Int,N}) where {T,N}
    v = nothing
    wanted_rank = find_rank(x, I...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == wanted_rank
        J = ntuple(idim-> shift_localindicies( I[idim], localindices(x)[idim]), N)
        v = getindex(x.p, J...)
    end
    v = MPI.bcast(v, wanted_rank, MPI.COMM_WORLD)
    v
end

function Base.setindex!(x::DevitoMPIAbstractArray{T,N}, v::T, I::Vararg{Int,N}) where {T,N}
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)
    if myrank == 0
        @warn "`setindex!` for Devito MPI Arrays has suboptimal performance. consider using `copy!`"
    end
    wanted_rank = find_rank(x, I...)
    if wanted_rank == 0
        received_v = v
    else
        message_tag = 2*MPI.Comm_size(MPI.COMM_WORLD)
        source_rank = 0
        send_mesg = [v]
        recv_mesg = 0 .* send_mesg
        rreq = ( myrank == wanted_rank ? MPI.Irecv!(recv_mesg, source_rank, message_tag, MPI.COMM_WORLD) : MPI.Request())
        sreq = ( myrank == source_rank ?  MPI.Isend(send_mesg, wanted_rank, message_tag, MPI.COMM_WORLD) : MPI.Request() )
        stats = MPI.Waitall!([rreq, sreq])
        received_v = recv_mesg[1]
    end
    if myrank == wanted_rank
        J = ntuple(idim-> shift_localindicies( I[idim], localindices(x)[idim]), N)
        setindex!(x.p, received_v, J...)
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

#
# Dimension
#
abstract type AbstractDimension end

# here is Devito's dimension type hierarchy:
# https://github.com/devitocodes/devito/blob/02bbefb7e380d299a2508fef2923c1a4fbd5c59d/devito/types/dimension.py

# Python <-> Julia quick-and-dirty type/struct for dimensions 
for (M,F) in ((:devito,:SpaceDimension), 
              (:devito,:SteppingDimension), 
              (:devito,:TimeDimension), 
              (:devito,:ConditionalDimension),
              (:devito,:Dimension),
              (:devito,:DefaultDimension))
    @eval begin
        struct $F <: AbstractDimension
            o::PyObject
        end
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x)
        $F(args...; kwargs...) = pycall($M.$F, $F, args...; kwargs...)
        export $F
    end
end

# subdimensions, generated using subdimension helper functions for simplicity 
abstract type AbstractSubDimension <: AbstractDimension end
for (M,F,G) in ((:devito,:SubDimensionLeft,:left), 
                (:devito,:SubDimensionRight, :right), 
                (:devito,:SubDimensionMiddle, :middle))
    @eval begin
        struct $F <: AbstractSubDimension
            o::PyObject
        end
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x) 
        $F(args...; kwargs...) = pycall($M.SubDimension.$G, $F, args...; kwargs...)
        export $F
    end
end

"""
    SubDimensionLeft(args...; kwargs...)

Creates middle a SubDimension.  Equivalent to devito.SubDimension.left helper function.

# Example
```julia
x = SpaceDimension(name="x")
xm = SubDimensionLeft(name="xl", parent=x, thickness=2)
```
"""
function SubDimensionLeft end

"""
    SubDimensionRight(args...; kwargs...)

Creates right a SubDimension.  Equivalent to devito.SubDimension.right helper function.

# Example
```julia
x = SpaceDimension(name="x")
xr = SubDimensionRight(name="xr", parent=x, thickness=3)
```
"""
function SubDimensionRight end

"""
    SubDimensionMiddle(args...; kwargs...)

Creates middle a SubDimension.  Equivalent to devito.SubDimension.middle helper function.

# Example
```julia
x = SpaceDimension(name="x")
xm = SubDimensionMiddle(name="xm", parent=x, thickness_left=2, thickness_right=3)
```
"""
function SubDimensionMiddle end

"""
    thickness(x::AbstractSubDimension)

Returns a tuple of a tuple containing information about the left and right thickness of a SubDimension and a symbol corresponding to each side's thickness.

# Example
```julia
x = SpaceDimension(name="x")
xr = SubDimensionRight(name="xr", parent=x, thickness=2)
thickness(xr)
```
"""
thickness(x::AbstractSubDimension) = x.o.thickness

"""
    parent(x::AbstractSubDimension)

Returns the parent dimension of a subdimension.

# Example
```julia
x = SpaceDimension(name="x")
xr = SubDimensionRight(name="xr", parent=x, thickness=2)
parent(xr)
````
"""
Base.parent(x::AbstractSubDimension) = x.o.parent

# Python <-> Julia quick-and-dirty type/struct mappings
for (M,F) in ((:devito,:Eq), (:devito,:Injection), (:devito, :Inc))

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

Base.:(==)(x::Eq,y::Eq) = x.o == y.o

struct Operator
    o::PyObject

    function Operator(args...; kwargs...)
        if :name ∈ keys(kwargs)
            new(pycall(devito.Operator, PyObject, args...; kwargs...))
        else
            new(pycall(devito.Operator, PyObject, args...; name="Kernel", kwargs...))
        end
    end
    
    function Operator(op::PyObject)
        if (:apply ∈ propertynames(op)) && (:ccode ∈ propertynames(op))
            new(op)
        else
            error("PyObject is not an operator")
        end
    end
    
end
PyCall.PyObject(x::Operator) = x.o
Base.convert(::Type{Operator}, x::PyObject) = Operator(x)
export Operator

"""
    Operator(expressions...[; optional named arguments])

Generate, JIT-compile and run C code starting from an ordered sequence of symbolic expressions,
and where you provide a list of `expressions` defining the computation.

# Optional named arguments
* `name::String` Name of the Operator, defaults to “Kernel”.
* `subs::Dict` Symbolic substitutions to be applied to expressions.
* `opt::String` The performance optimization level. Defaults to configuration["opt"].
* `language::String` The target language for shared-memory parallelism. Defaults to configuration["language"].
* `platform::String` The architecture the code is generated for. Defaults to configuration["platform"].
* `compiler::String` The backend compiler used to jit-compile the generated code. Defaults to configuration["compiler"].
"""
function Operator end

struct Constant{T}
    o::PyObject
end

"""
    Constant(args...; kwargs...)

Symbol representing a constant, scalar value in symbolic equations. 
A Constant carries a scalar value.

# kwargs
* `name::String` Name of the symbol.
* `value::Real` Value associated with the symbol.  Defaults to 0.
* `dtype::Type{AbstractFloat}` choose from `Float32` or `Float64`.  Default is `Float32`
"""
function Constant(args...; kwargs...)
    o =  pycall(devito.Constant, PyObject, args...; kwargs...)
    T = numpy_eltype(o.dtype)
    Constant{T}(o)
end

function Constant(o::PyObject)
    if (:is_const ∈ propertynames(o) ) && (o.is_const)
        T = numpy_eltype(o.dtype)
        Constant{T}(o)
    else
        error("PyObject is not a Constant")
    end
end

PyCall.PyObject(x::Constant{T}) where {T} = x.o
Base.convert(::Type{Constant}, x::PyObject) = Constant(x)

"""
    data(x::Constant{T})

Returns `value(x::Constant{T})`.  See `value` documentation for more information.
"""
data(x::Constant) = value(x)

"""
    value(x::Constant)

Returns the value of a devito constant. Can not be used to change constant value, for that use value!(x,y)
"""
value(x::Constant{T}) where {T} = convert(T,x.o._value)

"""
    isconst(x::Constant)

True if the symbol value cannot be modified within an Operator (and thus its value is provided by the user directly from Python-land), False otherwise.
"""
Base.isconst(x::Constant) = x.o.is_const

"""
    value!(x::Constant{T},y::T)

Change the numerical value of a constant, x, after creation to y, after converting y to datatype T of constant x.
"""
function value!(x::Constant{T},y::Real) where {T}
    x.o.data = PyObject(convert(T,y))
end

"""
    SpaceDimension(;kwargs...)

Construct a space dimension that defines the extend of a physical grid.

See https://www.devitoproject.org/devito/dimension.html?highlight=spacedimension#devito.types.dimension.SpaceDimension.

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
````
"""
function SpaceDimension end

Base.:(==)(x::AbstractDimension,y::AbstractDimension) = x.o == y.o
PyCall.PyObject(x::AbstractDimension) = x.o
"""
    Opertor(expressions; kwargs...)

Generate, JIT-compile and run C code starting from an ordered sequence of symbolic expressions.

See: https://www.devitoproject.org/devito/operator.html?highlight=operator#devito.operator.operator.Operator""

# Example
Assuming that one has constructed the following Devito expressions: `stencil_p`, `src_term` and `rec_term`,
```julia
op = Operator([stencil_p, src_term, rec_term]; name="opIso")
```
"""
function Operator end

"""
    ConditionalDimension(;kwargs)

Symbol defining a non-convex iteration sub-space derived from a parent Dimension, implemented by the compiler generating conditional “if-then” code within the parent Dimension’s iteration space.

See: https://www.devitoproject.org/devito/dimension.html?highlight=conditional#devito.types.dimension.ConditionalDimension

# Example 
```julia
size, factor = 16, 4
i  = SpaceDimension(name="i")
grid = Grid(shape=(size,),dimensions=(i,))
ci = ConditionalDimension(name="ci", parent=i, factor=factor)
g  = Devito.Function(name="g", grid=grid, shape=(size,), dimensions=(i,))
f  = Devito.Function(name="f", grid=grid, shape=(div(size,factor),), dimensions=(ci,))
op = Operator([Eq(g, 1), Eq(f, g)],name="Cond")
```
"""
function ConditionalDimension end


factor(x::ConditionalDimension) = x.o.factor
export factor
Base.parent(x::Union{ConditionalDimension,SteppingDimension}) = x.o.parent

#
# Grid
#
struct Grid{T,N}
    o::PyObject
end

"""
    Grid(; shape[, optional key-word arguments...])

Construct a grid that can be used in the construction of a Devito Functions.

See: https://www.devitoproject.org/devito/grid.html?highlight=grid#devito.types.Grid

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z), # z is fast (row-major)
    shape = (251,501),
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)
```
"""
function Grid(args...; kwargs...)
    o = pycall(devito.Grid, PyObject, args...; reversedims(kwargs)...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    Grid{T,N}(o)
end

PyCall.PyObject(x::Grid) = x.o

Base.:(==)(x::Grid{T,N},y::Grid{T,N}) where{T,N} = x.o == y.o
Base.size(grid::Grid{T,N}) where {T,N} = reverse((grid.o.shape)::NTuple{N,Int})
extent(grid::Grid{T,N}) where {T,N} = reverse((grid.o.extent)::NTuple{N,Float64})

"""
    origin(grid)

returns the tuple corresponding to the grid's origin
"""
origin(grid::Grid{T,N}) where {T,N} = reverse((grid.o.origin)::NTuple{N,Float64})
size_with_halo(grid::Grid{T,N}, h) where {T,N} = ntuple(i->size(grid)[i] + h[i][1] + h[i][2], N)
Base.size(grid::Grid, i::Int) = size(grid)[i]
Base.ndims(grid::Grid{T,N}) where {T,N} = N
Base.eltype(grid::Grid{T}) where {T} = T

spacing(x::Grid{T,N}) where {T,N} = reverse(x.o.spacing)
spacing_map(x::Grid{T,N}) where {T,N} = Dict( key => convert( T, val) for (key, val) in pairs(PyDict(x.o."spacing_map")))

#
# SubDomain
#

abstract type AbstractSubDomain{N} end

struct SubDomain{N} <: AbstractSubDomain{N}
    o::PyObject
end

PyCall.PyObject(x::AbstractSubDomain) = x.o

"""
    subdomains(grid)

returns subdomains associated with a Devito grid
"""
function subdomains(x::Grid{T,N}) where {T,N}
    dictpre =  PyDict(x.o."subdomains")
    dict = Dict()
    for key in keys(dictpre)
        dict[key] = SubDomain{N}(dictpre[key])
    end
    return dict
end

"""
    interior(x::grid)

returns the interior subdomain of a Devito grid
"""
interior(x::Grid{T,N}) where {T,N} = SubDomain{N}(x.o.interior)

Base.:(==)(x::AbstractSubDomain,y::AbstractSubDomain) = x.o == y.o

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

ismpi_distributed(o::PyObject) = (o._distributor === nothing) || (o._distributor.nprocs == 1) ? DevitoMPIFalse : DevitoMPITrue 

"""
    Devito.Function(; kwargs...)

Tensor symbol representing a discrete function in symbolic equations.

See: https://www.devitoproject.org/devito/function.html?highlight=function#devito.types.Function

# Example
```
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z),
    shape = (251,501), # assume x is first, z is second (i.e. z is fast in python)
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)

b = Devito.Function(name="b", grid=grid, space_order=8)
```
"""
function Function(args...; allowpro=true, kwargs...)
    if allowpro
        o = pycall(devitopro.Function, PyObject, args...; reversedims(kwargs)...)
    else
        o = pycall(devito.Function, PyObject, args...; reversedims(kwargs)...)
    end
    T = numpy_eltype(o.dtype)
    N = length(o.dimensions)
    M = ismpi_distributed(o)
    Function{T,N,M}(o)
end

function Function(o::PyObject)
    # ensure pyobject corresponds to a devito function
    isafunction = (:is_Function ∈ propertynames(o)) && (o.is_Function == true)
    isatimefunction = ((:is_TimeFunction ∈ propertynames(o)) && (o.is_TimeFunction == true))
    isasparsefunction = ((:is_SparseFunction ∈ propertynames(o)) && (o.is_SparseFunction == true))
    if (isafunction && ~(isatimefunction || isasparsefunction))
        T = numpy_eltype(o.dtype)
        N = length(o.dimensions)
        M = ismpi_distributed(o)
        return Function{T,N,M}(o)
    else
        error("PyObject is not a devito.Function")
    end
end

struct SubFunction{T,N,M} <: DiscreteFunction{T,N,M}
    o::PyObject
end

struct TimeFunction{T,N,M} <: DiscreteFunction{T,N,M}
    o::PyObject
end

"""
    TimeFunction(; kwargs...)

Tensor symbol representing a discrete function in symbolic equations.

See https://www.devitoproject.org/devito/timefunction.html?highlight=timefunction#devito.types.TimeFunction.

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z),
    shape = (251,501), # assume x is first, z is second (i.e. z is fast in python)
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)

p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
```
"""
function TimeFunction(args...; allowpro=true, serialization=nothing, kwargs...)
    local o
    if allowpro
        if serialization === nothing
            o = pycall(devitopro.TimeFunction, PyObject, args...; reversedims(kwargs)...)
        else
            if ~has_devitopro()
                @error "Automatic serialization only supported with devito pro"
            end
            # this is inelegant, TODO: find better way to handle layers.  
            # Issue is that PyCall interpets the layers as tuple, eliminating key metadata.
            # TODO: Generate MFE and submit as issue to PyCall
            py"""
            import devitopro
            def serializedtimefunc(serialization, **kwargs):
                return devitopro.TimeFunction(layers=devitopro.types.enriched.Disk, serialization=serialization, **kwargs)
            """
            o = py"serializedtimefunc"(serialization; Devito.reversedims(kwargs)...)
        end
    else
        o = pycall(devito.TimeFunction, PyObject, args...; reversedims(kwargs)...)
    end
    T = numpy_eltype(o.dtype)
    N = length(o.dimensions)
    M = ismpi_distributed(o)
    TimeFunction{T,N,M}(o)
end

function serial2str(x::TimeFunction)
    mypath = ""
    if hasproperty(x.o, :_fnbase)
        mypath = py"str"(x.o._fnbase)
    else
        @warn "Object doesn't have serialized path!"
    end
    return mypath
end

function str2serial(y::String)
    py"""
    from pathlib import Path
    def str2path(y):
        return Path(y)
    """
    return py"str2path"(y)
end


abstract type SparseDiscreteFunction{T,N,M} <:  DiscreteFunction{T,N,M} end

struct SparseTimeFunction{T,N,M} <: SparseDiscreteFunction{T,N,M}
    o::PyObject
end

"""
    SparseTimeFunction(; kwargs...)

Tensor symbol representing a space- and time-varying sparse array in symbolic equations.

See: https://www.devitoproject.org/devito/sparsetimefunction.html?highlight=sparsetimefunction

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z),
    shape = (251,501), # assume x is first, z is second (i.e. z is fast in python)
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)

time_range = 0.0f0:0.5f0:1000.0f0
src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=length(time_range))
```
"""
function SparseTimeFunction(args...; kwargs...)
    o = pycall(devito.SparseTimeFunction, PyObject, args...; reversedims(kwargs)...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    M = ismpi_distributed(o)
    SparseTimeFunction{T,N,M}(o)
end

function SparseTimeFunction(o::PyObject)
    if (:is_SparseTimeFunction ∈ propertynames(o)) && (o.is_SparseTimeFunction == true)
        T = numpy_eltype(o.dtype)
        N = length(o.shape)
        M = ismpi_distributed(o)
        return SparseTimeFunction{T,N,M}(o)
    else
        error("PyObject is not a devito.SparseTimeFunction")
    end     
end

struct SparseFunction{T,N,M} <: SparseDiscreteFunction{T,N,M}
    o::PyObject
end

"""
    SparseFunction(; kwargs...)

Tensor symbol representing a sparse array in symbolic equations.

See: https://www.devitoproject.org/devito/sparsefunction.html

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z),
    shape = (251,501), # assume x is first, z is second (i.e. z is fast in python)
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)

src = SparseFunction(name="src", grid=grid, npoint=1)
```
"""
function SparseFunction(args...; kwargs...)
    o = pycall(devito.SparseFunction, PyObject, args...; reversedims(kwargs)...)
    T = numpy_eltype(o.dtype)
    N = length(o.shape)
    M = ismpi_distributed(o)
    SparseFunction{T,N,M}(o)
end

function SparseFunction(o::PyObject)
    if ((:is_SparseFunction ∈ propertynames(o)) && (o.is_SparseFunction == true)) && ~((:is_SparseTimeFunction ∈ propertynames(o)) && (o.is_SparseTimeFunction == true))
        T = numpy_eltype(o.dtype)
        N = length(o.shape)
        M = ismpi_distributed(o)
        return SparseFunction{T,N,M}(o)
    else
        error("PyObject is not a devito.SparseFunction")
    end     
end

function CoordSlowSparseFunction(args...; kwargs...)
    @pydef mutable struct coordslowsparse <: devito.SparseFunction
    
        _sparse_position = 0
    end
    return SparseFunction(coordslowsparse(args...; reversedims(kwargs)...))
end

PyCall.PyObject(x::DiscreteFunction) = x.o

"""
    grid(f::DiscreteFunction)

Return the grid corresponding to the discrete function `f`.
"""
grid(x::Function{T,N}) where {T,N} = Grid{T,N}(x.o.grid)
grid(x::TimeFunction{T,N}) where {T,N} = Grid{T,N-1}(x.o.grid)

function grid(x::SparseDiscreteFunction{T}) where {T}
    N = length(x.o.grid.shape)
    Grid{T,N}(x.o.grid)
end

"""
    halo(x::DiscreteFunction)

Return the Devito "outer" halo size corresponding to the discrete function `f`.
"""
halo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o.halo)::NTuple{N,Tuple{Int,Int}}

"""
    inhalo(x::DiscreteFunction)

Return the Devito "inner" halo size used for domain decomposition, and corresponding to
the discrete function `f`.
"""
inhalo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o._size_inhalo)::NTuple{N,Tuple{Int,Int}}

"""
    size(x::DiscreteFunction)

Return the shape of the grid for the discrete function `x`.
"""
Base.size(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o.shape)::NTuple{N,Int}

"""
    ndims(x::DiscreteFunction)

Return the number of dimensions corresponding to the discrete function `x`.
"""
Base.ndims(x::DiscreteFunction{T,N}) where {T,N} = N

"""
    size_with_halo(x::DiscreteFunction)

Return the size of the grid associated with `x`, inclusive of the Devito "outer" halo.
"""
size_with_halo(x::DiscreteFunction{T,N}) where{T,N} = reverse(convert.(Int, x.o.shape_with_halo))::NTuple{N,Int}

"""
    size_with_inhalo(x::DiscreteFunction)

Return the size of the grid associated with `z`, inclusive the the Devito "inner" and "outer" halos.
"""
size_with_inhalo(x::DiscreteFunction{T,N}) where {T,N} = reverse(x.o._shape_with_inhalo)::NTuple{N,Int}

Base.size(x::SparseDiscreteFunction{T,N,DevitoMPITrue}) where {T,N} = size(data(x))

size_with_halo(x::SparseDiscreteFunction) = size(x)

localmask(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_domain[N-i+1].start)+1:convert(Int,x.o._mask_domain[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}
localmask_with_halo(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_outhalo[N-i+1].start)+1:convert(Int,x.o._mask_outhalo[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}
localmask_with_inhalo(x::DiscreteFunction{T,N}) where {T,N} = ntuple(i->convert(Int,x.o._mask_inhalo[N-i+1].start)+1:convert(Int,x.o._mask_inhalo[N-i+1].stop), N)::NTuple{N,UnitRange{Int}}

localindices(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask(x)
localindices_with_halo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask_with_halo(x)
localindices_with_inhalo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = localmask_with_inhalo(x)

"""
    space_order(x::Union{TimeFunction,Function})

Returns the space order for spatial derivatives defined on the associated TimeFunction or Function
"""
space_order(x::Union{TimeFunction,Function}) = x.o.space_order

"""
    forward(x::TimeFunction)

Returns the symbol for the time-forward state of the `TimeFunction`.

See: https://www.devitoproject.org/devito/timefunction.html?highlight=forward#devito.types.TimeFunction.forward
"""
forward(x::TimeFunction) = x.o.forward

"""
    backward(x::TimeFunction)

Returns the symbol for the time-backward state of the `TimeFunction`.

See: https://www.devitoproject.org/devito/timefunction.html?highlight=forward#devito.types.TimeFunction.backward
"""
backward(x::TimeFunction) = x.o.backward

"""
    time_dim(x::Union{Grid,TimeFunction})

Returns the time dimension for the associated object.
"""
time_dim(x::Union{Grid,TimeFunction}) = dimension(x.o.time_dim)

"""
    stepping_dim(x::Grid)

Returns the stepping dimension for the associated grid.
"""
stepping_dim(x::Grid) = dimension(x.o.stepping_dim)
export time_dim, stepping_dim

"""
    subs(f::DiscreteFunction{T,N,M},dict::Dict)

Perform substitution on the dimensions of Devito Discrete Function f based on a dictionary.

# Example
```julia
    grid = Grid(shape=(10,10,10))
    z,y,x = dimensions(grid)
    f = Devito.Function(grid=grid, name="f", staggered=x)
    subsdict = Dict(x=>x-spacing(x)/2)
    g = subs(f,subsdict)
```
"""
subs(f::DiscreteFunction{T,N,M},dict::Dict) where {T,N,M} = f.o.subs(dict)

"""
    evaluate(x::PyObject)

Evaluate a PyCall expression
"""
evaluate(x::PyObject) = x.evaluate

"""
    data(x::DiscreteFunction)

Return the data associated with the grid that corresponds to the discrete function `x`.  This is the
portion of the grid that excludes the halo.  In the case of non-MPI Devito, this returns an array
of type `DevitoArray`.  In the case of the MPI Devito, this returns an array of type `DevitoMPIArray`.

The `data` can be converted to an `Array` via `convert(Array, data(x))`.  In the case where `data(x)::DevitoMPIArray`,
this also *collects* the data onto MPI rank 0.
"""
data(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(DevitoArray{T,N}(x.o."_data_allocated"), localindices(x)...)

"""
    data_with_halo(x::DiscreteFunction)

Return the data associated with the grid that corresponds to the discrete function `x`.  This is the
portion of the grid that excludes the inner halo and includes the outer halo.  In the case of non-MPI
Devito, this returns an array of type `DevitoArray`.  In the case of the MPI Devito, this returns an
array of type `DevitoMPIArray`.

The `data` can be converted to an `Array` via `convert(Array, data(x))`.  In the case where `data(x)::DevitoMPIArray`,
this also *collects* the data onto MPI rank 0.
"""
data_with_halo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(DevitoArray{T,N}(x.o."_data_allocated"), localindices_with_halo(x)...)

"""
    data_with_inhalo(x::DiscreteFunction)

Return the data associated with the grid that corresponds to the discrete function `x`.  This is the
portion of the grid that includes the inner halo and includes the outer halo.  In the case of non-MPI
Devito, this returns an array of type `DevitoArray`.  In the case of the MPI Devito, this returns an
array of type `DevitoMPIArray`.

The `data` can be converted to an `Array` via `convert(Array, data(x))`.  In the case where `data(x)::DevitoMPIArray`,
this also *collects* the data onto MPI rank 0.
"""
data_with_inhalo(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = view(data_allocated(x), localindices_with_inhalo(x)...)

"""
    data_allocated(x::DiscreteFunction)

Return the data associated with the grid that corresponds to the discrete function `x`.  This is the
portion of the grid that includes the inner halo and includes the outer halo.  We expect this to be
equivalent to `data_with_inhalo`.

The `data` can be converted to an `Array` via `convert(Array, data(x))`.  In the case where `data(x)::DevitoMPIArray`,
this also *collects* the data onto MPI rank 0.
"""
data_allocated(x::DiscreteFunction{T,N,DevitoMPIFalse}) where {T,N} = DevitoArray{T,N}(x.o."_data_allocated")

function localindices(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    localinds = PyCall.trygetproperty(x.o,"local_indices",nothing)
    if localinds === nothing
        return ntuple(i -> 0:-1, N)
    else
        return ntuple(i->convert(Int,localinds[N-i+1].start)+1:convert(Int,localinds[N-i+1].stop), N)
    end
end

function one_based_decomposition(decomposition)
    for idim = 1:length(decomposition)
        if decomposition[idim] !== nothing
            for ipart = 1:length(decomposition[idim])
                decomposition[idim][ipart] .+= 1
            end
        end
    end
    decomposition
end

function getdecomp(x::DiscreteFunction)
    decomppre = reverse(x.o._decomposition)
    funcshape = reverse(x.o.shape)
    decompout = ()
    # if the decomp at a level is nothing, replace it with decomp over whole dim
    for i in 1:length(decomppre)
        if decomppre[i] === nothing
            decompout = (decompout..., ([0:funcshape[i]-1;],))
        else
            decompout = (decompout..., decomppre[i])
        end
    end
    return decompout
end

function getdecompwithhalo(x::DiscreteFunction)
    decomppre = reverse(x.o._decomposition_outhalo)
    funcshape = reverse(x.o.shape_with_halo)
    decompout = ()
    # if the decomp at a level is nothing, replace it with decomp over whole dim
    for i in 1:length(decomppre)
        if decomppre[i] === nothing
            decompout = (decompout..., ([0:funcshape[i]-1;],))
        else
            decompout = (decompout..., decomppre[i])
        end
    end
    return decompout
end

function topology(x::DiscreteFunction)
    # this checks for non-distributor dimensions and tacks them on in the right position
    distributordims = reverse(x.o._distributor.dimensions)
    functiondims    = reverse(x.o.dimensions)
    topopre = reverse(x.o._distributor.topology)
    topoout = ()
    j = 1
    for i in 1:length(functiondims)
        if (j <= length(distributordims)) && (functiondims[i] == distributordims[j])
            topoout = (topoout..., topopre[j])
            j = j+1
        else
            topoout = (topoout..., 1)
        end
    end
    return topoout
end

function mycoords(x::DiscreteFunction)
    # this checks for non-distributor dimensions and tacks them on in the right position
    distributordims = reverse(x.o._distributor.dimensions)
    functiondims    = reverse(x.o.dimensions)
    mycoordspre = reverse(x.o._distributor.mycoords) .+ 1
    mycoordsout = ()
    j = 1
    for i in 1:length(functiondims)
        if (j <= length(distributordims)) && (functiondims[i] == distributordims[j])
            mycoordsout = (mycoordsout..., mycoordspre[j])
            j = j+1
        else
            mycoordsout = (mycoordsout..., 1)
        end
    end
    return mycoordsout
end

decomposition(x::DiscreteFunction) = one_based_decomposition(getdecomp(x))
decomposition_with_halo(x::DiscreteFunction) = one_based_decomposition(getdecompwithhalo(x))

function decomposition_with_inhalo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    _decomposition = getdecomp(x)
    h = inhalo(x)
    ntuple(
        idim->begin
            if _decomposition[idim] === nothing
                nothing
            else
                M = length(_decomposition[idim])
                ntuple(
                    ipart->begin
                        n = length(_decomposition[idim][ipart])
                        strt = _decomposition[idim][ipart][1] + (h[idim][1] + h[idim][2])*(ipart-1) + 1
                        stop = _decomposition[idim][ipart][end] + (h[idim][1] + h[idim][2])*ipart + 1
                        [strt:stop;]
                    end,
                    M
                )
            end
        end,
        N
    )
end

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

function data(x::Function{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask(x)...)
    d = decomposition(x)
    t = topology(x)
    idxs = localindices(x)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,typeof(p),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_with_halo(x::Function{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_halo(x)...)
    d = decomposition_with_halo(x)
    t = topology(x)
    idxs = localindices_with_halo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,typeof(p),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_with_inhalo(x::Function{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_inhalo(x)...)
    d = decomposition_with_inhalo(x)
    t = topology(x)
    idxs = localindices_with_inhalo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,typeof(p),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_allocated(x::Function{T,N,DevitoMPITrue}) where {T,N}
    DevitoMPIArray{T,N}(x.o."_data_allocated", localindices_with_inhalo(x), decomposition(x), topology(x))
end

function data(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask(x)...)
    d = decomposition(x)
    t = topology(x)
    idxs = localindices(x)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,typeof(p),length(t),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_with_halo(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_halo(x)...)
    d = decomposition_with_halo(x)
    t = topology(x)
    idxs = localindices_with_halo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,typeof(p),length(t),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_with_inhalo(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_inhalo(x)...)
    d = decomposition_with_inhalo(x)
    t = topology(x)
    idxs = localindices_with_inhalo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,typeof(p),length(t),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function data_allocated(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    DevitoMPITimeArray{T,N}(x.o."_data_allocated", localindices_with_inhalo(x), decomposition(x), topology(x))
end

function data_allocated(x::SubFunction{T,2,DevitoMPITrue}) where {T}
    topo = (1, MPI.Comm_size(MPI.COMM_WORLD)) # topo is not defined for sparse decompositions
    d = DevitoMPIArray{T,2}(x.o."_data_allocated", localindices(x), decomposition(x), topo)
end

sparsetopo(x::Union{SparseFunction{T,N,DevitoMPITrue},SparseTimeFunction{T,N,DevitoMPITrue}}) where {T,N} = ntuple(i-> length(decomposition(x)[i]) > 1 ? MPI.Comm_size(MPI.COMM_WORLD) : 1, N)

localindxhelper(x) = length(x) > 1 ? x[MPI.Comm_rank(MPI.COMM_WORLD)+1] : x[1]

sparseindices(x::Union{SparseFunction{T,N,DevitoMPITrue},SparseTimeFunction{T,N,DevitoMPITrue}}) where {T,N} = localindxhelper.(decomposition(x))

function data_with_inhalo(x::SparseFunction{T,N,DevitoMPITrue}) where {T,N}
    d = DevitoMPISparseArray{T,N}(x.o."_data_allocated", sparseindices(x), decomposition(x), sparsetopo(x))
    MPI.Barrier(MPI.COMM_WORLD)
    d
end

# TODO - needed? <--
function data_with_inhalo(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N}
    d = DevitoMPISparseTimeArray{T,N}(x.o."_data_allocated", sparseindices(x), decomposition(x), sparsetopo(x))
    MPI.Barrier(MPI.COMM_WORLD)
    d
end

function data_with_inhalo(x::SparseDiscreteFunction{T,N,DevitoMPIFalse}) where {T,N}
    d = DevitoArray{T,N}(x.o."_data_allocated")
    d
end

data_with_halo(x::SparseDiscreteFunction{T,N,M}) where {T,N,M} = data_with_inhalo(x)
data(x::SparseDiscreteFunction{T,N,M}) where {T,N,M} = data_with_inhalo(x)
data(x::SubFunction{T,N,M}) where {T,N,M} = data_allocated(x)
# -->

"""
    coordinates(x::SparseDiscreteFunction)

Returns a Devito function associated with the coordinates of a sparse time function.
Note that contrary to typical Julia convention, coordinate order is from slow-to-fast (Python ordering).
Thus, for a 3D grid, the sparse time function coordinates would be ordered x,y,z.
"""
coordinates(x::SparseDiscreteFunction{T,N,M}) where {T,N,M} = SubFunction{T,2,M}(x.o.coordinates)

"""
    coordinates_data(x::SparseDiscreteFunction)

Returns a Devito array associated with the coordinates of a sparse time function.
Note that contrary to typical Julia convention, coordinate order is from slow-to-fast (Python ordering).
Thus, for a 3D grid, the sparse time function coordinates would be ordered x,y,z.
"""
coordinates_data(x::SparseDiscreteFunction{T,N,M}) where {T,N,M} = data(coordinates(x))

export DevitoArray, localindices, SubFunction
function dimension(o::PyObject)
    if :is_Dimension ∈ propertynames(o)
        if o.is_Conditional
            return ConditionalDimension(o)
        elseif o.is_Stepping
            return SteppingDimension(o)
        elseif o.is_Space
            return SpaceDimension(o)
        elseif o.is_Time
            return TimeDimension(o)
        elseif o.is_Default
            return DefaultDimension(o)
        elseif o.is_Dimension
            return Dimension(o)
        end
    end
    error("not implemented")
end

"""
    dimensions(x::Union{Grid,DiscreteFunction})

Returns a tuple with the dimensions associated with the Devito grid.
"""
function dimensions(x::Union{Grid{T,N},DiscreteFunction{T,N},AbstractSubDomain{N}}) where {T,N}
    ntuple(i->dimension(x.o.dimensions[N-i+1]), N)
end

"""
    inject(x::SparseDiscreteFunction; kwargs...)

Generate equations injecting an arbitrary expression into a field.

See: Generate equations injecting an arbitrary expression into a field.

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (x,z),
    shape = (251,501), # assume x is first, z is second (i.e. z is fast in python)
    origin = (0.0,0.0),
    extent = (1250.0,2500.0),
    dtype = Float32)

time_range = 0.0f0:0.5f0:1000.0f0
src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=length(time_range))
src_term = inject(src; field=forward(p), expr=2*src)
```
"""
inject(x::SparseDiscreteFunction, args...; kwargs...) = pycall(PyObject(x).inject, Injection, args...; kwargs...)

"""
    interpolate(x::SparseDiscreteFunction; kwargs...)

Generate equations interpolating an arbitrary expression into self.

See: https://www.devitoproject.org/devito/sparsetimefunction.html#devito.types.SparseTimeFunction.interpolate

# Example
```julia
x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))
grid = Grid(
    dimensions = (z,x),
    shape = (501,251), # assume z is first, x is second
    origin = (0.0,0.0),
    extent = (2500.0,1250.0),
    dtype = Float32)

p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)

time_range = 0.0f0:0.5f0:1000.0f0
nz,nx,δz,δx = size(grid)...,spacing(grid)...
rec = SparseTimeFunction(name="rec", grid=grid, npoint=nx, nt=length(time_range))
rec_coords = coordinates_data(rec)
rec_coords[1,:] .= 10.0
rec_coords[2,:] .= δx*(0:nx-1)


rec_term = interpolate(rec, expr=p)
```
"""
interpolate(x::SparseDiscreteFunction; kwargs...) = pycall(PyObject(x).interpolate, PyObject; kwargs...)

"""
apply(    operator::Operator; kwargs...)

Execute the Devito operator, `Operator`.

See: https://www.devitoproject.org/devito/operator.html?highlight=apply#devito.operator.operator.Operator.apply

Note that this returns a `summary::Dict` of the action of applying the operator.  This contains information
such as the number of floating point operations executed per second.
"""
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

# metaprograming for various derivatives
for F in (:dx,:dy,:dz,:dxr,:dyr,:dzr,:dxl,:dyl,:dzl)
    @eval begin
        $F(x::Union{DiscreteFunction,PyObject}, args...; kwargs...) = ( hasproperty(PyObject(x),Symbol($F)) ? pycall(PyObject(x).$F, PyObject, args...; kwargs...) : PyObject(0) )
        $F(x::Union{Constant,Number}, args...; kwargs...) = PyObject(0)
        export $F
    end
end
"""
    dx(f::Union{DiscreteFunction,PyObject,Constant,Number}, args...; kwargs...)

Returns the symbol for the first derivative with respect to x if f is a Function with dimension x.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dx end

"""
    dy(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first derivative with respect to yif f is a Function with dimension y.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dy end

"""
    dz(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first derivative with respect to zif f is a Function with dimension z.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""

function dz end

"""
    dxl(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first backward one-sided derivative with respect to x if f is a Function with dimension x.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dxl end

"""
    dyl(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first backward one-sided derivative with respect to y if f is a Function with dimension y.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dyl end

"""
    dzl(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first backward one-sided derivative with respect to z if f is a Function with dimension y.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dzl end

"""
    dxr(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first forward one-sided derivative with respect to x if f is a Function with dimension x.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dxr end

"""
    dyr(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first forward one-sided derivative with respect to y if f is a Function with dimension y.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dyr end

"""
    dzr(f::DiscreteFunction, args...; kwargs...)

Returns the symbol for the first forward one-sided derivative with respect to z if f is a Function with dimension z.
Otherwise returns 0.  Thus, the derivative of a function with respect to a dimension it doesn't have is zero, as is the derivative of a constant.
"""
function dz end

# metaprograming for various derivatives
for F in (:dt,:dt2)
    @eval begin
        $F(x::Union{TimeFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).$F, PyObject, args...; kwargs...)
        export $F
    end
end

"""
    dt(f::TimeFunction, args...; kwargs...)

Returns the symbol for the first time derivative of a time function
"""
function dt end

"""
    dt2(f::TimeFunction, args...; kwargs...)

Returns the symbol for the second time derivative of a time function
"""
function dt2 end

# metaprogramming for basic operations
for F in ( :+, :-, :*, :/, :^)
    @eval begin
        Base.$F(x::Real,y::Union{DiscreteFunction,Constant,AbstractDimension}) = $F(PyObject(x),PyObject(y))
        Base.$F(x::Union{DiscreteFunction,Constant,AbstractDimension}, y::Union{DiscreteFunction,Constant,AbstractDimension}) = $F(PyObject(x),PyObject(y))
        Base.$F(x::Union{DiscreteFunction,Constant,Dimension}, y::PyObject) = $F(x.o,y)
        Base.$F(x::PyObject, y::Union{DiscreteFunction,Constant,AbstractDimension}) = $F(x,y.o)
        Base.$F(x::Union{DiscreteFunction,Constant,AbstractDimension}, y::Real) = $F(PyObject(x),PyObject(y))
    end
end

Base.:(-)(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = -1*x
Base.:(+)(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = x

# metaprogramming to access Devito dimension boolean attributes
for F in (:is_Dimension, :is_Space, :is_Time, :is_Default, :is_Custom, :is_Derived, :is_NonlinearDerived, :is_Sub, :is_Conditional, :is_Stepping, :is_Modulo, :is_Incr)
    @eval begin
        $F(x::AbstractDimension) = x.o.$F::Bool
        export $F
    end
end
# metaprogramming for devito conditionals
for (M,F) in ((:devito,:Ne),(:devito,:Gt),(:devito,:Ge),(:devito,:Lt),(:devito,:Le))
    @eval begin
        $F(x::Union{Real,DiscreteFunction,PyObject,AbstractDimension},y::Union{Real,DiscreteFunction,PyObject,AbstractDimension}) = $M.$F(PyObject(x),PyObject(y))
        export $F
    end
end

# metaprogramming for symbolic operations on Devito dimensions
for F in (:symbolic_min, :symbolic_max, :spacing, :symbolic_size)
    @eval begin
        $F(x::AbstractDimension) = PyObject(x).$F
        export $F
    end
end

"""
    symbolic_min(x::Dimension)

Symbol defining the minimum point of the Dimension
"""
function symbolic_min end

"""
    symbolic_max(x::Dimension)

Symbol defining the maximum point of the Dimension
"""
function symbolic_max end

"""
    spacing(x::Dimension)

Symbol representing the physical spacing along the Dimension.
"""
function spacing end

"""
    is_Derived(x::Dimension)

Returns true when dimension is derived, false when it is not
"""
function is_Derived end

"""
    symbolic_size(x::Dimension)

Symbol defining the size of the Dimension
"""
function symbolic_size end

# metaprograming for Devito functions taking variable number of arguments
for (M,F) in ((:devito,:Min), (:devito,:Max),(:sympy,:And))
    @eval begin
        $F(args...) = $M.$F((PyObject.(args))...)
        export $F
    end
end

"""
    Min(args...)

Can be used in a Devito.Eq to return the minimum of a collection of arguments
Example:
```julia
    eqmin = Eq(f,Min(g,1))
```
Is equivalent to f = Min(g,1) for Devito functions f,g
"""
function Min end

"""
    Max(args...)

Can be used in a Devito.Eq to return the minimum of a collection of arguments
Example:
```julia
    eqmax = Eq(f,Max(g,1))
```
Is equivalent to f = Max(g,1) for Devito functions f,g
"""
function Max end

# metaprograming for Devito mathematical operations ( more exist and may be added as required, find them at https://github.com/devitocodes/devito/blob/a8a33dc55ac3be008644c58a76b671028625679a/devito/finite_differences/elementary.py )

# these are broken out into four groups to help keep track of how they behave for unit testing

# functions defined on real numbers with equivalent in base
for F in (:cos, :sin, :tan, :sinh, :cosh, :tanh, :exp, :floor)
    @eval begin
        Base.$F(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = devito.$F(PyObject(x))
    end
end
# functions defined on real numbers who are written differently in base
for F in (:Abs,:ceiling)
    @eval begin
        $F(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = devito.$F(PyObject(x))
        export $F
    end
end
# functions defined on positive numbers with equivalent in base
for F in (:sqrt,)
    @eval begin
        Base.$F(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = devito.$F(PyObject(x))
    end
end
# functions defined on positive numbers who are written differently in base
for F in (:ln,)
    @eval begin
        $F(x::Union{AbstractDimension,DiscreteFunction,PyObject,Constant}) = devito.$F(PyObject(x))
        export $F
    end
end

""" 
    Mod(x::AbstractDimension,y::Int)

Perform Modular division on a dimension
"""
Mod(x::AbstractDimension,y::Int) = sympy.Mod(PyObject(x),PyObject(y))
export Mod

"""Get symbolic representation for function index object"""
function Base.getindex(x::Union{TimeFunction,Function},args...)
    py"""
    def indexobj(x,*args):
        return x[args]
    """
   return py"indexobj"(x,reverse(args)...)
end

# helper functions for mapping arguments to python
shiftarg(x::Int) = x-1
shiftarg(x) = x

function pygetindex(x::PyObject,args...)
    py"""
    def indexobj(x,*args):
        return x[args]
    """
   return py"indexobj"(x,reverse(shiftarg.(args))...)
end

struct IndexedData
    o::PyObject
end

"""
The wrapped IndexedData object.
"""
indexed(x::DiscreteFunction) = IndexedData(x)
IndexedData(x::DiscreteFunction) = IndexedData(x.o.indexed)
PyCall.PyObject(x::IndexedData) = x.o

Base.getindex(x::IndexedData,args...) = Indexed(pygetindex(x.o, args...))

struct Indexed
    o::PyObject
    Indexed(o) = ( hasproperty(o, :is_Indexed) && getproperty(o, :is_Indexed) ? new(o) : error("not indexed"))
end

PyCall.PyObject(x::Indexed) = x.o

"""
    ccode(x::Operator; filename="")

Print the ccode associated with a devito operator.  
If filename is provided, writes ccode to disk using that filename
"""
function ccode(x::Operator; filename="")
    py"""
    def ccode(x, filename):
        if filename == "":
            return print(x)
        else:
            with open(filename, 'w') as f:
                print(x,file=f)
    """
   py"ccode"(x.o,filename)
   return nothing
end

"""
    SubDomain(name, instructions)

Create a subdomain by passing a list of instructions for each dimension.
Using an instruction with (nothing,) implies that the whole dimension should be used for that subdomain, as will ("middle",0,0)

Examples:
```julia
instructions = ("left",2),("middle",3,3)
SubDomain("subdomain_name",instructions)
```
or 
```julia
instructions = [("right",4),("middle",1,2)]
SubDomain("subdomain_name",instructions)
```
or
```julia
SubDomain("subdomain_name",("right",2),("left",1))
```
"""
SubDomain(name::String, instructions::Vector) = SubDomain(name, instructions...)
SubDomain(name::String, instructions::Tuple{Vararg{Tuple}}) = SubDomain(name, instructions...)
function SubDomain(name::String, instructions...)
    # copy and reverse instructions
    instructions = reverse(instructions)
    N = length(instructions)
    @pydef mutable struct subdom <: devito.SubDomain
        function __init__(self, name, instructions)
            self.name = name
            self.instructions = instructions
        end
        function define(self, dimensions)
            dims = PyDict()
            for idim = 1:length(dimensions)
                # dims[dimensions[idim]] = self.instructions[idim] in ((nothing,),("middle",0,0)) ? dimensions[idim] : self.instructions[idim]
                dims[dimensions[idim]] = self.instructions[idim] in (nothing,) ? dimensions[idim] : self.instructions[idim]
            end
            return dims
        end
    end
    return SubDomain{N}(subdom(name,instructions))    
end

struct Buffer
    o::PyObject
end

"""
    Buffer(value::Int)
Construct a devito buffer.  This may be used as a save= keyword argument in the construction of TimeFunctions.
"""
Buffer(value::Int) = Buffer(pycall(devito.Buffer, PyObject, value))
PyCall.PyObject(x::Buffer) = x.o

"""
    nsimplify(expr::PyObject; constants=(), tolerance=none, full=false, rational=none, rational_conversion="base10")

Wrapper around `sympy.nsimplify`.
Find a simple representation for a number or, if there are free symbols or
if ``rational=True``, then replace Floats with their Rational equivalents. If
no change is made and rational is not False then Floats will at least be
converted to Rationals.

# Explanation
For numerical expressions, a simple formula that numerically matches the
given numerical expression is sought (and the input should be possible
to evalf to a precision of at least 30 digits).
Optionally, a list of (rationally independent) constants to
include in the formula may be given.
A lower tolerance may be set to find less exact matches. If no tolerance
is given then the least precise value will set the tolerance (e.g. Floats
default to 15 digits of precision, so would be tolerance=10**-15).
With ``full=True``, a more extensive search is performed
(this is useful to find simpler numbers when the tolerance
is set low).
When converting to rational, if rational_conversion='base10' (the default), then
convert floats to rationals using their base-10 (string) representation.
When rational_conversion='exact' it uses the exact, base-2 representation.

See https://github.com/sympy/sympy/blob/52f606a503cea5e9588de14150ccb9f7f9ed4752/sympy/simplify/simplify.py .

# Examples:
```julia
nsimplify(π) # PyObject 314159265358979/100000000000000
```
```julia
nsimplify(π; tolerance=0.1) # PyObject 22/7
```
"""
nsimplify(expr::PyObject; constants=(), tolerance=nothing, full=false, rational=nothing, rational_conversion="base10") = pycall(sympy.nsimplify, PyObject, expr, constants=constants, tolerance=tolerance, full=full, rational=rational, rational_conversion=rational_conversion)

nsimplify(x::Number; kwargs...) = nsimplify(PyObject(x); kwargs...)

"""
    solve(eq::PyObject, target::PyObject; kwargs...)

Algebraically rearrange an Eq w.r.t. a given symbol.
This is a wrapper around ``devito.solve``, which in turn is a wrapper around ``sympy.solve``.

# Parameters
* `eq::PyObject` expr-like. The equation to be rearranged.
* `target::PyObject` The symbol w.r.t. which the equation is rearranged. May be a `Function` or any other symbolic object.

## kwargs
* Symbolic optimizations applied while rearranging the equation. For more information. refer to `sympy.solve.__doc__`.
"""
solve(eq::PyObject, target::PyObject; kwargs...) = pycall(devito.solve, PyObject, eq, target, kwargs...)

"""
    name(x::Union{SubDomain, DiscreteFunction, TimeFunction, Function, Constant, AbstractDimension, Operator})

returns the name of the Devito object
"""
name(x::Union{SubDomain, DiscreteFunction, Constant, AbstractDimension, Operator}) = x.o.name

Base.isequal(x::Union{SubDomain, DiscreteFunction, Constant, AbstractDimension, Operator, Grid, Eq, Inc, Injection, SparseDiscreteFunction}, y::Union{SubDomain, DiscreteFunction, Constant, AbstractDimension, Operator, Grid, Eq, Inc, Injection, SparseDiscreteFunction}) = isequal(PyObject(x), PyObject(y))

Base.hash(x::Union{SubDomain, DiscreteFunction, Constant, AbstractDimension, Operator, Grid, Eq, Inc, Injection}) = hash(PyObject(x))

# metaprogramming for unary ops
for F in (:Byref, :Deref, :Cast)
    @eval begin
        struct $F
            o::PyObject
        end
        $F(base::Union{DiscreteFunction,IndexedData,Indexed,String}, kwargs...) = pycall(devito.symbolics.$F, $F, base, kwargs...) # Todo: support Sympy.Basic as well
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x)
        export $F 
    end
end

"""
Symbolic representation of the C notation `&expr`.
"""
function Byref end

"""
Symbolic representation of the C notation `*expr`.
"""
function Deref end

"""
Symbolic representation of the C notation `(type)expr`.
"""
function Cast end

# metaprograming for various devito types for use in C
for F in (:Pointer,)
    @eval begin
        struct $F
            o::PyObject
        end
        $F(args...; kwargs...) = pycall(devito.types.$F, $F, args...; kwargs...)
        PyCall.PyObject(x::$F) = x.o
        Base.convert(::Type{$F}, x::PyObject) = $F(x)
        export $F 
    end
end

"""
Symbolic representation of a pointer in C
"""
function Pointer end

# DevitoPro Stuff

struct ABox{N} <: Devito.AbstractSubDomain{N}
    o::PyObject
end

function ABox(src::Union{Devito.SparseTimeFunction,Nothing}, rcv::Union{Devito.SparseTimeFunction,Nothing}, vp::Devito.Function{T,N}, space_order::Int; kwargs...) where {T,N}
    if ~has_devitopro()
        @error "ABox only supported with DevitoPro"
    end
    o = pycall(devitopro.ABox, PyObject, src, rcv, vp, space_order; kwargs...)
    ABox{N}(o)
end

intersection(box::ABox{N}, sub::Devito.SubDomain{N}) where {N} = ABox{N}(pycall(PyObject(box).intersection, PyObject, PyObject(sub)))

vp(abox::ABox) = Devito.Function(abox.o.vp)
eps(abox::ABox) = abox.o.eps
src(abox::ABox) = (typeof(abox.o.src) <: Nothing ? nothing : Devito.SparseTimeFunction(abox.o.src))
rcv(abox::ABox) = (typeof(abox.o.rcv) <: Nothing ? nothing : Devito.SparseTimeFunction(abox.o.rcv))
grid(abox::ABox) = Devito.grid(vp(abox))
function subdomains(abox::ABox{N}) where {N}
    dict = Dict()
    for dom in abox.o._subdomains
        dict[dom.name] = SubDomain{N}(dom)
    end
    return dict
end
compute(abox::ABox; dt) = abox.o._compute(; dt=dt)

export ABox

struct CCall
    o::PyObject
end

PyCall.PyObject(x::CCall) = x.o

function CCall(name::String; header=nothing, header_dirs = (), libs = (), lib_dirs = (), target = "host", types = ())
    if ~has_devitopro()
        @error "CCall only supported with DevitoPro"
    end
    classname = Symbol(uppercasefirst(name))
    @eval begin
        @pydef mutable struct $classname <: devitopro.CCall
            name = $name
            header = $header
            header_dirs = $header_dirs
            libs = $libs
            lib_dirs = $lib_dirs
            target = $target
            types = $types
        end
        return CCall($classname)
    end
end

name(x::CCall) = x.o.name
header(x::CCall) = x.o.header
header_dirs(x::CCall) = x.o.header_dirs
libs(x::CCall) = x.o.libs
lib_dirs(x::CCall) = x.o.lib_dirs
target(x::CCall) = x.o.target
types(x::CCall) = x.o.types

(f::CCall)(args...; kwargs...) = f.o(args...; kwargs...)

export CCall


export Buffer, Constant, CoordSlowSparseFunction, DiscreteFunction, Grid, Function, SparseFunction, SparseTimeFunction, SubDomain, TimeFunction, apply, backward, ccode, configuration, configuration!, coordinates, coordinates_data, data, data_allocated, data_with_halo, data_with_inhalo, dimension, dimensions, dx, dy, dz, evaluate, extent, forward, grid, halo, indexed, inject, interpolate, localindices, localindices_with_halo, localindices_with_inhalo, localsize, name, nsimplify, origin, size_with_halo, simplify, solve, space_order, spacing, spacing_map, step, subdomains, subs, thickness, value, value!

end
