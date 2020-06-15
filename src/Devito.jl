#=
1. setting earth model properties
   i. get size information (including) halo from grid.
   ii. get the size that includes the halo
   iii. get localindices ??
2. setting wavelets
=#

module Devito

using MPI, PyCall

const numpy = PyNULL()
const devito = PyNULL()
const seismic = PyNULL()

function __init__()
    copy!(numpy, pyimport("numpy"))
    copy!(devito, pyimport("devito"))
    copy!(seismic, pyimport("examples.seismic"))

py"""
def fill_function_from_array(x, value):
    x.data[:] = value[:]

def fill_function_from_number(x, value):
    x.data[:] = value
"""
end

# Devito configuration methods
function configuration!(key, value)
    c = PyDict(devito."configuration")
    c[key] = value
    c[key]
end
configuration(key) = PyDict(devito."configuration")[key]
configuration() = PyDict(devito."configuration")

# Python <-> Julia type/struct mappings
for (M,F) in (
        (:devito,:Constant), (:devito,:Eq), (:devito,:Function), (:devito,:Grid), (:devito,:Injection), (:devito,:Operator), (:devito,:SpaceDimension), (:devito,:TimeFunction),
        (:seismic, :Receiver), (:seismic,:RickerSource), (:seismic,:TimeAxis))
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

PyCall.PyObject(::Type{Float32}) = numpy.float32
PyCall.PyObject(::Type{Float64}) = numpy.float64

Base.copy!(x::Function, value::AbstractArray) = py"fill_function_from_array"(x, value)
Base.fill!(x::Function, value::Number) = py"fill_function_from_number"(x, value)

struct Dimension # TODO .. what is the corresponding python type?
    o::PyObject
end
PyCall.PyObject(x::Dimension) = x.o
Base.convert(::Type{Dimension}, o::PyObject) = Dimension(o)

# convenience methods
function dimensions(p::TimeFunction)
    x = PyObject(p).dimensions
    ntuple(i->convert(Dimension, x[i]), length(x))
end
spacing(x::Dimension) = (o = PyObject(x); o.spacing)

spacing_map(x::Grid) = PyDict(PyObject(x)."spacing_map")
Base.size(x::Grid) = PyObject(x).shape
Base.size(x::Grid, i) = PyObject(x).shape[i]
function Base.eltype(x::Grid)
    o = PyObject(x)
    if o.dtype == numpy.float32
        return Float32
    elseif o.dtype == numpy.float64
        return Float64
    else
        error("Grid element type is not recognized")
    end
end
spacing(x::Grid) = convert.(eltype(x), PyObject(x).spacing)
spacing(x::Grid, i) = convert(eltype(x), PyObject(x).spacing[i])

Base.step(x::TimeAxis) = PyObject(x).step

backward(x::TimeFunction) = PyObject(x).backward
forward(x::TimeFunction) = PyObject(x).forward
grid(x::TimeFunction) = PyObject(x).grid

inject(x::RickerSource, args...; kwargs...) = pycall(PyObject(x).inject, Injection, args...; kwargs...)

interpolate(x::Receiver; kwargs...) = pycall(PyObject(x).interpolate, PyObject; kwargs...)
data(x::Receiver) = PyObject(x).data

apply(x::Operator, args...; kwargs...) = pycall(PyObject(x).apply, PyObject, args...; kwargs...)

dx(x::Union{Function,TimeFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dx, PyObject, args...; kwargs...)
dy(x::Union{Function,TimeFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dy, PyObject, args...; kwargs...)
dz(x::Union{Function,TimeFunction,PyObject}, args...; kwargs...) = pycall(PyObject(x).dz, PyObject, args...; kwargs...)

lindices(x::TimeFunction) = PyObject(x).local_indices

data_nompi(timefunction::TimeFunction) = PyObject(timefunction).data

function data_mpi_2D(timefunction::TimeFunction)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD

    indices = lindices(timefunction)
    x = data(timefunction)
    y = zeros(size(x,1), eltype(grid(timefunction)), size(grid(timefunction)))

    indices_1 = indices[1].start+1:indices[1].stop
    indices_2 = indices[2].start+1:indices[2].stop

    y[:,indices_1,indices_2] .= x
    MPI.Reduce(y, +, 0, comm)
end

function data_mpi_3D(timefunction::TimeFunction)
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD

    indices = lindices(timefunction)
    x = data(timefunction)
    y = zeros(size(x,1), eltype(grid(timefunction)), size(grid(timefunction)))

    # TODO ... generalize this
    indices_1 = indices[1].start+1:indices[1].stop
    indices_2 = indices[2].start+1:indices[2].stop
    indices_3 = indices[3].start+1:indices[3].stop

    y[:,indices_1,indices_2,indices_3] .= x
    MPI.Reduce(y, +, 0, comm)
end

# TODO: use parametric types for TimeFunction to make this type stable
function data(timefunction::TimeFunction)
    local d
    if configuration("mpi") == 0
        d = data_nompi(timefunction)
    else
        grd = grid(timefunction)
        if ndims(grid(timefunction)) == 2
            d = data_mpi_2D(timefunction)
        elseif ndims(grid(timefunction)) == 3
            d = data_mpi_3D(timefunction)
        else
            error("grid with MPI and ndims=$(ndims(grd)) is not supported.")
        end
    end
    d
end

Base.:*(x::Function, y::PyObject) = PyObject(x)*y
Base.:*(x::PyObject, y::Function) = x*PyObject(y)
Base.:/(x::Function, y::PyObject) = PyObject(x)/y
Base.:/(x::PyObject, y::Function) = x/PyObject(y)
Base.:^(x::Function, y) = PyObject(x)^y

export apply, backward, configuration, configuration!, data, dimensions, dx, dy, dz, forward, interpolate, inject, spacing, spacing_map, step

end
