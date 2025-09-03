ENV["OMP_NUM_THREADS"] = "24"

# This demo implements a 3D acoustic wave equation with spatially variable density and Q
# using a second-order time discretization and eighth-order space discretization.
# The wave equation is solved using a finite-difference time-domain (FDTD) method.
# A Ricker wavelet source is injected into the model, and the wavefield is recorded
# at a set of receiver locations. The wavefield in the model and at the receivers
# is plotted with PyPlot.

# Note: the Devito decoupler requires DevitoPro

using Devito, PyPlot

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")

function ricker(f, _t, t₀)
    t = reshape(_t, length(_t), 1)
    (1.0 .- 2.0 * (pi * f * (t .- t₀)).^2) .* exp.(-(pi * f * (t .- t₀)).^2)
end

function model(; decoupler)
    configuration!("decoupler", decoupler)

    write(stdout, "inside model()\n")
    grid = Grid(
        shape = (121,121,121),
        origin = (0.0,0.0,0.0),
        extent = (1210.0,1210.0,1210.0),
        dtype = Float32)

    b = Devito.Function(name="b", grid=grid, space_order=8)
    v = Devito.Function(name="v", grid=grid, space_order=8)
    q = Devito.Function(name="woverq", grid=grid, space_order=8)

    b_data = data_with_halo(b)
    v_data = data_with_halo(v)
    q_data = data_with_halo(q)

    b_data .= 1
    v_data .= 1.5
    q_data .= 1/1000

    time_range = 0.0f0:1.0f0:750.0f0

    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
    z,y,x,t = dimensions(p)

    src = SparseTimeFunction(name="src", grid=grid, f0=0.01f0, npoint=1, nt=length(time_range), coordinates=[605.0 605.0 10.0])
    src_data = data(src)

    w = zeros(Float32, 1, length(time_range))
    w[1,:] .= ricker(0.01, collect(time_range), 125)[:]
    copy!(src_data, w)

    src_term = inject(src; field=forward(p), expr=src * spacing(t)^2 * v^2 / b)

    nz,ny,nx,δz,δy,δx = size(grid)...,spacing(grid)...
    rec = SparseTimeFunction(name="rec", grid=grid, npoint=nx, nt=length(time_range))
    rec_coords = coordinates_data(rec)
    _rec_coords = zeros(Float32, length(dimensions(p))-1, nx)
    _rec_coords[1,:] .= δx*(0:nx-1)
    _rec_coords[2,:] .= 605
    _rec_coords[3,:] .= 20
    copy!(rec_coords, _rec_coords)
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

    # update the dimension spacing_map to include the time dimension
    # these symbols will be replaced with the relevant scalars by the Operator
    dt = step(time_range)
    smap = spacing_map(grid)
    smap[spacing(t)] = dt

    op = Operator([stencil_p, src_term, rec_term], subs=smap, name="OpExampleIso")
    @elapsed summary = apply(op)
    @show summary

    _p = data(p)
    _d = data(rec)

    figure(figsize=(8,8))
    imshow(_p[:,:,60,1],aspect="auto",cmap="gray",clim=.500*[-1,1]*maximum(abs,_p))
    xlabel("X")
    ylabel("Z")
    title("3D Wavefield in the model")
    tight_layout()
    savefig("image-p-3d-decoupler$(decoupler).png", dpi=100)

    figure(figsize=(8,8))
    imshow(_d,aspect="auto",cmap="gray",clim=.010*[-1,1]*maximum(abs,_d))
    xlabel("Receiver")
    ylabel("Time")
    title("3D Receiver Wavefield")
    tight_layout()
    savefig("image-d-3d-decoupler$(decoupler).png", dpi=100)

    nothing
end

model(; decoupler=0)

# Note: the Devito decoupler requires DevitoPro
model(; decoupler=1)
