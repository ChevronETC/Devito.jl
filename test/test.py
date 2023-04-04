import devito, devitopro

def ezpz(shape,time_M):
    g = devito.Grid(shape=shape)
    u = devito.TimeFunction(grid=g, name="u")
    eq = devito.Eq(u.forward,u+2)
    op = devito.Operator([eq], name="ezpz")
    op.apply(time_m=0, time_M=time_M)
    print(u.data.min(), u.data.max())


shape = (4,4,4)
time_M = 3

ezpz(shape, time_M)
