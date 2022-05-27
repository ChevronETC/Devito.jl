using Revise

using Devito

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", false)

function ricker(f, _t, t₀)
    t = reshape(_t, length(_t), 1)
    (1.0 .- 2.0 * (pi * f * (t .- t₀)).^2) .* exp.(-(pi * f * (t .- t₀)).^2)
end

grid = Grid(
    shape = (121,121,121),
    origin = (0.0,0.0,0.0),
    extent = (1210.0,1210.0,1210.0),
    dtype = Float32)

b = Devito.Function(name="b", grid=grid, space_order=8)
v = Devito.Function(name="v", grid=grid, space_order=8)
q = Devito.Function(name="woverq", grid=grid, space_order=8)

b_data = data(b)
v_data = data(v)
q_data = data(q)

# note that the folowing is faster if we include the halo
# i.e. use the data_with_halo method rather than the data method.
b_data .= 1
v_data .= 1.5
q_data .= 1/1000

time_range = 0.0f0:1.0f0:750.0f0

p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
z,y,x,t = dimensions(p)

src = SparseTimeFunction(name="src", grid=grid, f0=0.01f0, npoint=1, nt=length(time_range), coordinates=[605.0 605.0 10.0])
src_data = data(src)
w = ricker(0.01, collect(time_range), 125)
copy!(src_data, w)
src_term = inject(src; field=forward(p), expr=src*spacing(t)^2*v^2/b)

nz,ny,nx,δz,δy,δx = size(grid)...,spacing(grid)...
rec = SparseTimeFunction(name="rec", grid=grid, npoint=nx, nt=length(time_range))
rec_coords = coordinates_data(rec)
rec_coords[1,:] .= δx*(0:nx-1)
rec_coords[2,:] .= 605
rec_coords[3,:] .= 20
rec_term = interpolate(rec, expr=p)

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

dt = step(time_range)
smap = spacing_map(grid)
smap[spacing(t)] = dt

op = Operator([stencil_p, src_term, rec_term], subs=smap, name="OpExampleIso")

summary = apply(op)

using PyPlot

_p = data_with_halo(p)
extrema(_p)

d = data(rec)
extrema(d)

figure();imshow(_p[:,:,60,1])
display(gcf())

figure();imshow(d, aspect="auto",cmap="gray",clim=.1*[-1,1]*maximum(abs,d))
display(gcf())