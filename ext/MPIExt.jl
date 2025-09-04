module MPIExt

using MPI

using Devito
using Devito.PyCall
using Devito.Strided

import Devito: DiscreteFunction, TimeFunction, SparseFunction, SparseTimeFunction, SubFunction, SparseDiscreteFunction
import Devito: DevitoMPITrue, Function, inhalo, size_with_inhalo, halo, mycoords, topology, decomposition, parent
import Devito: localmask, localmask_with_halo, localmask_with_inhalo, decomposition_with_halo
import Devito: localindices, localindices_with_halo, localindices_with_inhalo
import Devito: data_allocated, data, data_with_halo, data_with_inhalo

abstract type DevitoMPIAbstractArray{T,N} <: AbstractArray{T,N} end

Base.parent(x::DevitoMPIAbstractArray) = x.p
localsize(x::DevitoMPIAbstractArray{T,N}) where {T,N} = ntuple(i->size(x.local_indices[i])[1], N)
localindices(x::DevitoMPIAbstractArray{T,N}) where {T,N} = x.local_indices
decomposition(x::DevitoMPIAbstractArray) = x.decomposition
topology(x::DevitoMPIAbstractArray) = x.topology


function _size_from_local_indices(local_indices::NTuple{N,UnitRange{Int64}}) where {N}
    n = Devito.ntuple(i->(size(local_indices[i])[1] > 0 ? local_indices[i][end] : 0), N)
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
        _y = Devito.convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
    else
        _y = Array{T,N}(undef, ntuple(_->0, N))
    end
    _y
end


function Base.copyto!(dst::DevitoMPIArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    _counts = counts(dst)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = Devito.copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
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
        _y = Devito.convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
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
        _y = Devito.copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
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
        _y = Devito.convert_resort_array!(Array{T,N}(undef, size(x)), y, x.topology, x.decomposition)
    else
        _y = Array{T,N}(undef, ntuple(_->0, N))
    end
    _y
end

function Base.copyto!(dst::Union{DevitoMPISparseTimeArray{T,N},DevitoMPISparseArray{T,N}}, src::Array{T,N}) where {T,N}
    _counts = counts(dst)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        _y = Devito.copyto_resort_array!(Vector{T}(undef, length(src)), src, dst.topology, dst.decomposition)
        data_vbuffer = VBuffer(_y, _counts)
    else
        data_vbuffer = VBuffer(nothing)
    end
    _dst = MPI.Scatterv!(data_vbuffer, Vector{T}(undef, _counts[MPI.Comm_rank(MPI.COMM_WORLD)+1]), 0, MPI.COMM_WORLD)
    copyto!(parent(dst), _dst)
end


function find_rank(x::DevitoMPIAbstractArray{T,N}, I::Vararg{Int,N}) where {T,N}
    decomp = decomposition(x)
    rank_position = Devito.in_range.(I,decomp)
    helper = Devito.helix_helper(topology(x))
    rank = sum((rank_position .- 1) .* helper)
    return rank
end


function Base.getindex(x::DevitoMPIAbstractArray{T,N}, I::Vararg{Int,N}) where {T,N}
    v = nothing
    wanted_rank = find_rank(x, I...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == wanted_rank
        J = ntuple(idim-> Devito.shift_localindicies( I[idim], localindices(x)[idim]), N)
        v = getindex(x.p, J...)
    end
    v = MPI.bcast(v, wanted_rank, MPI.COMM_WORLD)
    v
end

# 2025-09-03 JKW this is never ever used in practice - remove?
# function Base.setindex!(x::DevitoMPIAbstractArray{T,N}, v::T, I::Vararg{Int,N}) where {T,N}
#     myrank = MPI.Comm_rank(MPI.COMM_WORLD)
#     if myrank == 0
#         @warn "`setindex!` for Devito MPI Arrays has suboptimal performance. consider using `copy!`"
#     end
#     wanted_rank = find_rank(x, I...)
#     if wanted_rank == 0
#         received_v = v
#     else
#         message_tag = 2*MPI.Comm_size(MPI.COMM_WORLD)
#         source_rank = 0
#         send_mesg = [v]
#         recv_mesg = 0 .* send_mesg
#         rreq = ( myrank == wanted_rank ? MPI.Irecv!(recv_mesg, source_rank, message_tag, MPI.COMM_WORLD) : MPI.Request())
#         sreq = ( myrank == source_rank ?  MPI.Isend(send_mesg, wanted_rank, message_tag, MPI.COMM_WORLD) : MPI.Request() )
#         stats = MPI.Waitall!([rreq, sreq])
#         received_v = recv_mesg[1]
#     end
#     if myrank == wanted_rank
#         J = ntuple(idim-> Devito.shift_localindicies( I[idim], localindices(x)[idim]), N)
#         setindex!(x.p, received_v, J...)
#     end
#     MPI.Barrier(MPI.COMM_WORLD)
# end

Base.size(x::SparseDiscreteFunction{T,N,DevitoMPITrue}) where {T,N} = size(data(x))

function Devito.data(x::Function{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask(x)...)
    d = decomposition(x)
    t = topology(x)
    idxs = localindices(x)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,typeof(p),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function Devito.data_with_halo(x::Function{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_halo(x)...)
    d = decomposition_with_halo(x)
    t = topology(x)
    idxs = localindices_with_halo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPIArray{T,N,typeof(p),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function Devito.data_with_inhalo(x::Function{T,N,DevitoMPITrue}) where {T,N}
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

function Devito.data(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask(x)...)
    d = decomposition(x)
    t = topology(x)
    idxs = localindices(x)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,typeof(p),length(t),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function Devito.data_with_halo(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
    p = sview(parent(data_allocated(x)), localmask_with_halo(x)...)
    d = decomposition_with_halo(x)
    t = topology(x)
    idxs = localindices_with_halo(x)
    n = _size_from_local_indices(idxs)
    DevitoMPITimeArray{T,N,typeof(p),length(t),typeof(d)}(x.o."_data_allocated", p, idxs, d, t, n)
end

function Devito.data_with_inhalo(x::TimeFunction{T,N,DevitoMPITrue}) where {T,N}
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

function Devito.data_with_inhalo(x::SparseFunction{T,N,DevitoMPITrue}) where {T,N}
    d = DevitoMPISparseArray{T,N}(x.o."_data_allocated", sparseindices(x), decomposition(x), sparsetopo(x))
    MPI.Barrier(MPI.COMM_WORLD)
    d
end

# TODO - needed? <--
function Devito.data_with_inhalo(x::SparseTimeFunction{T,N,DevitoMPITrue}) where {T,N}
    d = DevitoMPISparseTimeArray{T,N}(x.o."_data_allocated", sparseindices(x), decomposition(x), sparsetopo(x))
    MPI.Barrier(MPI.COMM_WORLD)
    d
end


function localindices(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    localinds = PyCall.trygetproperty(x.o,"local_indices",nothing)
    if localinds === nothing
        return ntuple(i -> 0:-1, N)
    else
        return ntuple(i->convert(Int,localinds[N-i+1].start)+1:convert(Int,localinds[N-i+1].stop), N)
    end
end


function decomposition_with_inhalo(x::DiscreteFunction{T,N,DevitoMPITrue}) where {T,N}
    _decomposition = Devito.getdecomp(x)
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

end