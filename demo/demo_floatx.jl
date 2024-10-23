using Devito


dtype = FloatX16(1.5f0, 4.5f0)
grid = Grid((11, 11))

f = Devito.Function(name="f", grid=grid, space_order=4, dtype=dtype)

@show f
@show typeof(data(f).p)

data(f) .= 1.5f0

@show data(f)[1,1]
@show Devito.decompress(data(f)[1,1])