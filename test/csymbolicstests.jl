using Devito, PythonCall, Test

const ctypes = pyimport("ctypes")

@testset "Devito Pointer" begin
    p = Pointer(name="pointer")
    @test pyconvert(Bool, getproperty(Py(p), :_C_ctype) == ctypes.c_void_p)
end

@testset "Devito Unary Ops" begin
    g = Grid(shape=(4,4))
    f = Devito.Function(name="f", grid=g)
    bref = Byref(f)
    @test pyconvert(String, getproperty(Py(bref), :_op)) == "&"
    dref = Deref(f)
    @test pyconvert(String, getproperty(Py(dref), :_op)) == "*"
    cst  = Cast(f, "char *")
    @test pyconvert(String, getproperty(Py(cst), :_op)) == "(char*)"
end
