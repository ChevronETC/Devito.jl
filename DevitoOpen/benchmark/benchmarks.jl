using Devito, BenchmarkTools

const SUITE = BenchmarkGroup()

nz,ny,nx = 102,101,100
grd = Grid(shape=(nz,ny,nx))
f = Devito.Function(name="f", grid=grd, space_order=8)
d = data(f)
dhalo = data_with_halo(f)
dinhalo = data_with_inhalo(f)
dalloc = data_allocated(f)

_d = zeros(eltype(d), size(d))
_dhalo = zeros(eltype(dhalo),  size(dhalo))
_dinhalo = zeros(eltype(dinhalo), size(dinhalo))
_dalloc = zeros(eltype(dalloc), size(dalloc))

SUITE["function"] = BenchmarkGroup()
SUITE["function"]["data"] = @benchmarkable data($f)
SUITE["function"]["data with halo"] = @benchmarkable data_with_halo($f)
SUITE["function"]["data with inhalo"] = @benchmarkable data_with_inhalo($f)
SUITE["function"]["data allocated"] = @benchmarkable data_allocated($f)
SUITE["function"]["data copy!"] = @benchmarkable copy!($_d, $d)
SUITE["function"]["data with halo copy!"] = @benchmarkable copy!($_dhalo, $dhalo)
SUITE["function"]["data with inhalo copy!"] = @benchmarkable copy!($_dinhalo, $dinhalo)
SUITE["function"]["data allocated copy!"] = @benchmarkable copy!($_dalloc, $dalloc)

stf = SparseTimeFunction(name="stf", grid=grd, npoint=1, nt=1000, coordinates=[50 50 50])
tf = TimeFunction(name="tf", grid=grd, time_order=8, space_order=8)

SUITE["sparse time function"] = BenchmarkGroup()
SUITE["sparse time function"]["data"] = @benchmarkable data($stf)
SUITE["sparse time function"]["data with halo"] = @benchmarkable data_with_halo($stf)
SUITE["sparse time function"]["data with inhalo"] = @benchmarkable data_with_inhalo($stf)
SUITE["sparse time function"]["data allocated"] = @benchmarkable data_allocated($stf)
SUITE["sparse time function"]["inject"] = @benchmarkable inject($stf; field=$(forward(tf)), expr=$stf)
SUITE["sparse time function"]["interpolate"] = @benchmarkable interpolate($stf; expr=tf)

SUITE
