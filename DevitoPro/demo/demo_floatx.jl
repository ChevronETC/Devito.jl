using Devito

dtype_f = FloatX16(+1.25f0, +1.75f0)
dtype_g = FloatX16(-1.75f0, -1.25f0)
grid = Grid((11, 11))

f = Devito.Function(name="f", grid=grid, space_order=4, dtype=dtype_f)
g = Devito.Function(name="g", grid=grid, space_order=4, dtype=dtype_g)

@show f
@show typeof(data(f).p)

data(f) .= +1.5f0
data(g) .= -1.5f0

@show data(f)[1,1]
@show Devito.decompress(data(f)[1,1])

@show data(g)[1,1]
@show Devito.decompress(data(g)[1,1])

nothing