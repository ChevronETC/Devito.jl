using Devito

grid = Grid(shape=(5,6))
f = Devito.Function(name="f", grid=grid)
g = Devito.Function(name="g", grid=grid, dtype=Devito.Float16(0,1))
h = Devito.Function(name="h", grid=grid, dtype=Devito.Float8(1, 5))