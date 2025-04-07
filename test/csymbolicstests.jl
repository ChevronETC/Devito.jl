using Devito, PyCall, Test

const ctypes = PyNULL()
copy!(ctypes, pyimport("ctypes"))

@testset "Devito Pointer" begin
    p = Pointer(name="pointer")
    @test getproperty(PyObject(p), :_C_ctype) == ctypes.c_void_p
end

@testset "Devito Unary Ops" begin
    g = Grid(shape=(4,4))
    f = Devito.Function(name="f", grid=g)
    bref = Byref(f)
    @test getproperty(PyObject(bref), :_op) == "&"
    dref = Deref(f)
    @test getproperty(PyObject(dref), :_op) == "*"
    cst  = Cast(f, "char *")
    @test getproperty(PyObject( cst), :_op) == "(char*)"
end
