# using PyCall

# @info "run"
# run(`python test.py`)

# @info "pyinclude"
# @pyinclude("test.py")

# @info "using Devito"
using Libdl
Libdl.dlopen("libgomp.so.1", Libdl.RTLD_GLOBAL)
using Devito, MPI
MPI.Init()
# Libdl.dlopen("libnvc.so", Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
# Libdl.dlopen("libnvcpumath.so", Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
# Libdl.dlopen("libssp.so.0", Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
# @info "$(Libdl.dllist())"
# @info "Julia approach"
g = Grid(shape=(60,70,80))
u = TimeFunction(name="u", grid=g)
eq = Eq(forward(u), u+1)
op = Operator([eq], name="ezpz")
apply(op,time_m=0, time_M=500)
myarray = convert(Array, data(u))
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @show extrema(myarray)
end

# @info "run"
# run(`python test.py`)

# @info "pyinclude"
# @pyinclude("test.py")