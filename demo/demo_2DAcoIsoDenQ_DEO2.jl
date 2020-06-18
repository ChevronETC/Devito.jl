using Revise

using Devito

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", 0)

x = SpaceDimension(name="x", spacing=Constant(name="h_x", value=5.0))
z = SpaceDimension(name="z", spacing=Constant(name="h_z", value=5.0))

grid = Grid(
    dimensions = (x,z),
    shape = (501,501),
    origin = (0.0,0.0,0.0),
    extent = (2500.0,2500.0),
    dtype = Float32)

b = Devito.Function(name="b", grid=grid, space_order=8)
v = Devito.Function(name="v", grid=grid, space_order=8)
q = Devito.Function(name="woverq", grid=grid, space_order=8)

copy!(b, ones(Float32,size(grid))) # alternative: fill!(b, 1)
copy!(q, (1/1000)*ones(Float32,size(grid))) # alternative: fill!(q, 1)
copy!(v, 1.5f0*ones(Float32,size(grid))) # alternative: fill!(v, 1.5)

time_range = TimeAxis(start=0.0f0, stop=500.0f0, step=1.0f0)

p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
t,x,z = dimensions(p)

src = RickerSource(name="src", grid=grid, f0=0.01f0, npoint=1, time_range=time_range,
    coordinates=[1250.0 5.0])
src_term = inject(src; field=PyObject(p).forward, expr=src*spacing(t)^2*v^2/b)

nz,nx,δz,δx = size(grid)...,spacing(grid)...
rec_coords = zeros(nx,2)
rec_coords[:,1] .= δx*[0:nx-1;]
rec_coords[:,2] .= 10.0
rec = Receiver(name="rec", grid=grid, npoint=nz, time_range=time_range, coordinates=rec_coords)
rec_term = interpolate(rec, expr=p)

g1(field) = dx(field,x0=x+spacing(x)/2)
g3(field) = dz(field,x0=z+spacing(z)/2)
g1_tilde(field) = dx(field,x0=x-spacing(x)/2)
g3_tilde(field) = dz(field,x0=z-spacing(z)/2)

update_p = spacing(t)^2 * v^2 / b *
    (g1_tilde(b * g1(p)) + g3_tilde(b * g3(p))) +
    (2 - spacing(t) * q) * p +
    (spacing(t) * q - 1) * backward(p)

stencil_p = Eq(forward(p), update_p)

dt = step(time_range)
smap = spacing_map(grid)
smap[spacing(t)] = dt

op = Operator([stencil_p, src_term, rec_term], subs=smap, name="OpExampleIso")

apply(op)

using PyPlot

_p = data(p)
extrema(_p)

d = data(rec)

figure();imshow(_p[1,:,:])
display(gcf())

figure();imshow(d, aspect="auto",cmap="gray",clim=.1*[-1,1]*maximum(abs,d))
display(gcf())
