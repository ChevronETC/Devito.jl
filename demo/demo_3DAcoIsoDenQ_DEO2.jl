using Revise

using Devito

configuration!("log-level", "CRITICAL")
configuration!("language", "openmp")
configuration!("mpi", 0)

grid = Grid(
    shape = (121,121,601),
    origin = (0.0,0.0,0.0),
    extent = (1210.0,1210.0,1210.0),
    dtype = Float32)

b = Devito.Function(name="b", grid=grid, space_order=8)
v = Devito.Function(name="v", grid=grid, space_order=8)
q = Devito.Function(name="woverq", grid=grid, space_order=8)

time_range = TimeAxis(start=0.0f0, stop=1250.0f0, step=1.0f0)

p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
t,x,y,z = dimensions(p)

src = RickerSource(name="src", grid=grid, f0=0.01f0, npoint=1, time_range=time_range,
    coordinates=[605.0 605.0 605.0])

nz,ny,nx,δz,δy,δx = size(grid)...,spacing(grid)...
rec_coords = zeros(size(grid,3),3)
rec_coords[:,1] .= 0.6*dx*(size(grid,1)-1)
rec = Receiver(name="rec", grid=grid, npoint=nz, time_range=time_range, coordinates=rec_coords)

src_term = inject(src; field=forward(p), expr=src*spacing(t)^2*v^2/b)

g1(field) = dx(field,x0=x+spacing(x)/2)
g2(field) = dy(field,x0=y+spacing(y)/2)
g3(field) = dz(field,x0=z+spacing(z)/2)
g1_tilde(field) = dx(field,x0=x-spacing(x)/2)
g2_tilde(field) = dy(field,x0=y-spacing(y)/2)
g3_tilde(field) = dz(field,x0=z-spacing(z)/2)

update_p = spacing(t)^2 * v^2 / b *
    (g1_tilde(b * g1(p)) + g2_tilde(b * g2(p)) + g3_tilde(b * g3(p))) +
    (2 - spacing(t) * q) * p +
    (spacing(t) * q - 1) * backward(p)

stencil_p = Eq(forward(p), update_p)

spacing_map(grid)[spacing(t)] = dt
spacing_map(grid)

dt = step(time_range)
smap = spacing_map(grid)
smap[spacing(t)] = dt

op = Operator([stencil_p, src_term], subs=smap, name="OpExampleIso")

apply(op)
