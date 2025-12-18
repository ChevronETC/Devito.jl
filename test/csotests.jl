using Devito, PythonCall, Test

function typedict()
    _typedict = Dict{DataType, Py}()
    _typedict[Float32] = Devito.numpy.float32
    _typedict[Float64] = Devito.numpy.float64
    _typedict[Int8] = Devito.numpy.int8
    _typedict[UInt8] = Devito.numpy.uint8
    _typedict[Int16] = Devito.numpy.int16
    _typedict[UInt16] = Devito.numpy.uint16
    _typedict[Int32] = Devito.numpy.int32
    _typedict[Int64] = Devito.numpy.int64
    _typedict[ComplexF32] = Devito.numpy.complex64
    _typedict[ComplexF64] = Devito.numpy.complex128
    _typedict
end

@testset "Exercise types T=$T" for T in (Float32, Float64, Int8, UInt8, Int16, UInt16, Int32, Int64, ComplexF32, ComplexF64)
    td = typedict()    
    @test pyconvert(Bool, Devito.numpy.dtype(td[T]) == Devito.numpy.dtype(T))
    @test T == Devito._numpy_eltype(td[T])
end

