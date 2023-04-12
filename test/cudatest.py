import devito, devitopro
import numpy as np 
from devitopro.types.enriched import NoLayers

compression = 'bitcomp'
dtype=np.float32
nt = 10
grid = devito.Grid(shape=(50, 50), dtype=dtype)

u = devitopro.TimeFunction(name='u', grid=grid)
v = devitopro.TimeFunction(name='v', grid=grid)
usave = devitopro.TimeFunction(name='usave', grid=grid, save=nt, space='local',
                        compression=compression, layers=NoLayers)
vsave = devitopro.TimeFunction(name='vsave', grid=grid, save=nt, layers=NoLayers)

# Forward operator (compress)
eqns0 = [
    devito.Eq(u.forward, u + 1.),  # PDE
    devito.Eq(usave, u)            # u.dt2*v
]
op0 = devito.Operator(eqns0, opt=('advanced'))

# Backward operator (decompress)
eqns1 = [
    devito.Eq(v, usave),    # Adjoint PDE
    devito.Eq(vsave, v),    # xcor
]
op1 = devito.Operator(eqns1, opt=('advanced'))

op0.apply(time_M=nt)
op1.apply(time_M=nt-1)