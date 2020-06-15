using Revise
using Distributed, AzManagers

ENV["OMP_NUM_THREADS"] = 4
addprocs("cbox120", 1; group="tqff-devito7", logname="tqff-devito7", mpi_ranks_per_worker=30)

@everywhere using Revise
@everywhere using Devito

@everywhere configuration!("log-level", "DEBUG")
@everywhere configuration!("language", "openmp")
@everywhere configuration!("mpi", 1)

@everywhere function model()
    write(stdout, "inside model()\n")
    grid = Grid(
        shape = (1201,1201,601),
        origin = (0.0,0.0,0.0),
        extent = (12000.0,12000.0,6000.0),
        dtype = Float32)

    write(stdout, "HERE1\n")
    b = Devito.Function(name="b", grid=grid, space_order=8)
    v = Devito.Function(name="vel", grid=grid, space_order=8)
    q = Devito.Function(name="wOverQ", grid=grid, space_order=8)
    write(stdout, "HERE2\n")

    copy!(b, ones(Float32,size(grid))) # alternative: fill!(b, 1)
    copy!(v, 1.5f0*ones(Float32,size(grid))) # alternative: fill!(v, 1.5)
    copy!(q, (1/1000)*ones(Float32,size(grid))) # alternative: fill!(q, 1)

    time_range = TimeAxis(start=0.0f0, stop=250.0f0, step=1.0f0)
    
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=8)
    t, x, y, z = dimensions(p)

    src = RickerSource(name="src", grid=grid, f0=0.01f0, npoint=1, time_range=time_range,
        coordinates=[6500.0 6500.0 10.0])
    src_term = inject(src; field=forward(p), expr=src * spacing(t)^2 * v^2 / b)

    nz,ny,nx,δz,δy,δx = size(grid)...,spacing(grid)...
    rec_coords = zeros(nx,3)
    rec_coords[:,1] .= δx*[0:nx-1;]
    rec_coords[:,2] .= 6500.0
    rec_coords[:,3] .= 20.0
    rec = Receiver(name="rec", grid=grid, npoint=nx, time_range=time_range, coordinates=rec_coords)
    rec_term = interpolate(rec, expr=p)

    g1(field) = dx(field,x0=x+spacing(x)/2)
    g2(field) = dy(field,x0=y+spacing(y)/2)
    g3(field) = dz(field,x0=z+spacing(z)/2)
    g1_tilde(field) = dx(field,x0=x-spacing(x)/2)
    g2_tilde(field) = dy(field,x0=y-spacing(y)/2)
    g3_tilde(field) = dz(field,x0=z-spacing(z)/2)
    
    # write the time update equation for u_x
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
    
    bx,by = 19,8
    t_apply = @elapsed apply(op; x0_blk0_size=bx, y0_blk0_size=by)

    write(stdout, "t_appy=$t_apply\n")

    data(p), data(rec)
end

p, d = model()