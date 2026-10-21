using Devito, PythonCall, Test
import PythonCall: pynew, pycopy!

const ctypes = pynew()
pycopy!(ctypes, pyimport("ctypes"))

@testset "Devito Pointer" begin
    p = Pointer(name="pointer")
    @test pyconvert(Bool, Py(p)._C_ctype == ctypes.c_void_p)
end

@testset "Devito Unary Ops" begin
    g = Grid(shape=(4,4))
    f = Devito.Function(name="f", grid=g)
    bref = Byref(f)
    @test pyconvert(String, Py(bref)._op) == "&"
    dref = Deref(f)
    @test pyconvert(String, Py(dref)._op) == "*"
    cst  = Cast(f, "char *")
    @test pyconvert(String, Py(cst)._op) == "(char*)"
end
