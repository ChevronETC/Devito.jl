using Devito, BenchmarkTools

const SUITE = BenchmarkGroup()

nx,ny,nz = 100,101,102
grid = Grid(shape=(nx,ny,nz))
f = Devito.Function(name="f", grid=grid, space_order=8)
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
