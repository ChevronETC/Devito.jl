import devito

def ezpz(shape,time_M):
    g = devito.Grid(shape=shape)
    u = devito.TimeFunction(grid=g, name="u")
    eq = devito.Eq(u.forward,u+1)
    op = devito.Operator([eq], name="ezpz")
    op.apply(time_m=0, time_M=time_M)
    print(u.data.min(), u.data.max())


shape = (60,70,80)
time_M = 600

ezpz(shape, time_M)
