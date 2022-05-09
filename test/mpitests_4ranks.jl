using Devito, LinearAlgebra, MPI, Random, Strided, Test

MPI.Init()
configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", true)

@testset "DevitoMPIArray, copy!, no halo, n=$n" for n in ( (11,10), (12,11,10))
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})
    @test size(b_data) == n
    b_data_test = zeros(Float32, n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test = reshape(Float32[1:prod(n);], n)
    end
    copy!(b_data, b_data_test)
    _b_data = data(b)
    b_data_test = reshape(Float32[1:prod(n);], n)

    for rnk in 0:3
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[1:6,1:5]
                    @test parent(b_data) ≈ b_data_test[1:6,1:5]
                else
                    @test parent(_b_data) ≈ b_data_test[:,1:6,1:5]
                    @test parent(b_data) ≈ b_data_test[:,1:6,1:5]
                end
                @test isa(parent(_b_data), StridedView)
            end
            if rnk == 1
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[7:11,1:5]
                    @test parent(b_data) ≈ b_data_test[7:11,1:5]
                else
                    @test parent(_b_data) ≈ b_data_test[:,7:11,1:5]
                    @test parent(b_data) ≈ b_data_test[:,7:11,1:5]
                end
                @test isa(parent(_b_data), StridedView)
            end
            if rnk == 2
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[1:6,6:10]
                    @test parent(b_data) ≈ b_data_test[1:6,6:10]
                else
                    @test parent(_b_data) ≈ b_data_test[:,1:6,6:10]
                    @test parent(b_data) ≈ b_data_test[:,1:6,6:10]
                end
                @test isa(parent(_b_data), StridedView)
            end
            if rnk == 3
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[7:11,6:10]
                    @test parent(b_data) ≈ b_data_test[7:11,6:10]
                else
                    @test parent(_b_data) ≈ b_data_test[:,7:11,6:10]
                    @test parent(b_data) ≈ b_data_test[:,7:11,6:10]
                end
                @test isa(parent(_b_data), StridedView)
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "Convert data from rank 0 to DevitoMPIArray then back, no halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)

    b_data_test = zeros(Float32, n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(n);], n)
    end
    copy!(b_data, b_data_test)
    _b_data = data(b)

    b_data_out = convert(Array, _b_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_out ≈ b_data_test
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

@testset "DevitoMPITimeArray, copy!, data, no halo, n=$n" for n in ( (11,10), (12,11,10))
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data(p)

    _n = length(n) == 2 ? (11,10,3) : (12,11,10,3)
    p_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        p_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(p_data, p_data_test)
    p_data_test .= reshape(Float32[1:prod(_n);], _n)

    p_data_local = parent(p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if length(n) == 2
            @test p_data_local ≈ p_data_test[1:6,1:5,:]
        else
            @test p_data_local ≈ p_data_test[:,1:6,1:5,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 1
        if length(n) == 2
            @test p_data_local ≈ p_data_test[7:11,1:5,:]
        else
            @test p_data_local ≈ p_data_test[:,7:11,1:5,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 2
        if length(n) == 2
            @test p_data_local ≈ p_data_test[1:6,6:10,:]
        else
            @test p_data_local ≈ p_data_test[:,1:6,6:10,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 3
        if length(n) == 2
            @test p_data_local ≈ p_data_test[7:11,6:10,:]
        else
            @test p_data_local ≈ p_data_test[:,7:11,6:10,:]
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

@testset "DevitoMPITimeArray coordinates check" begin
    ny,nx = 4,6

    grd = Grid(shape=(ny,nx), extent=(ny-1,nx-1), dtype=Float32)
    time_order = 1
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, save=time_order+1)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, save=time_order+1)
    sx = SparseTimeFunction(name="sx", grid=grd, npoint=ny*nx, nt=time_order+1)
    sy = SparseTimeFunction(name="sy", grid=grd, npoint=ny*nx, nt=time_order+1)

    cx = [ix-1 for iy = 1:ny, ix=1:nx][:]
    cy = [iy-1 for iy = 1:ny, ix=1:nx][:]

    coords = zeros(Float32, 2, ny*nx)
    coords[1,:] .= cx
    coords[2,:] .= cy
    copy!(coordinates(sx), coords)
    copy!(coordinates(sy), coords)

    datx = reshape(Float32[ix for iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny, time_order+1)
    daty = reshape(Float32[iy for iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny, time_order+1)

    copy!(data(sx), datx)
    copy!(data(sy), daty)

    eqx = inject(sx, field=forward(fx), expr=sx)
    eqy = inject(sy, field=forward(fy), expr=sy)
    op = Operator([eqx, eqy], name="CoordOp")
    apply(op)

    x = convert(Array, data(fx))
    y = convert(Array, data(fy))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if VERSION >= v"1.7"
            @test x ≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0]
            @test y ≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;; 1.0 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0 4.0]
        else
            _x = zeros(Float32, ny, nx, 2)
            _x[:,:,1] .= [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
            _x[:,:,2] .= [1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0; 1.0 2.0 3.0 4.0 5.0 6.0]
            @test x ≈ _x
            _y = zeros(Float32, ny, nx, 2)
            _y[:,:,1] .= [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
            _y[:,:,2] .= [1.0 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0 4.0]
            @test y ≈ _y
        end
    end
end

@testset "DevitoMPITimeArray coordinates check, 3D" begin
    nz,ny,nx = 4,5,6

    grd = Grid(shape=(nz,ny,nx), extent=(nz-1,ny-1,nx-1), dtype=Float32)
    time_order = 1
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, save=time_order+1)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, save=time_order+1)
    fz = TimeFunction(name="fz", grid=grd, time_order=time_order, save=time_order+1)
    sx = SparseTimeFunction(name="sx", grid=grd, npoint=nz*ny*nx, nt=time_order+1)
    sy = SparseTimeFunction(name="sy", grid=grd, npoint=nz*ny*nx, nt=time_order+1)
    sz = SparseTimeFunction(name="sz", grid=grd, npoint=nz*ny*nx, nt=time_order+1)

    cx = [ix-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]
    cy = [iy-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]
    cz = [iz-1 for iz = 1:nz, iy = 1:ny, ix=1:nx][:]

    coords = zeros(Float32, 3, nz*ny*nx)
    coords[1,:] .= cx
    coords[2,:] .= cy
    coords[3,:] .= cz
    copy!(coordinates(sx), coords)
    copy!(coordinates(sy), coords)
    copy!(coordinates(sz), coords)

    datx = reshape(Float32[ix for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)
    daty = reshape(Float32[iy for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)
    datz = reshape(Float32[iz for iz = 1:nz, iy = 1:ny, ix=1:nx, it = 1:time_order+1][:], nx*ny*nz, time_order+1)

    copy!(data(sx), datx)
    copy!(data(sy), daty)
    copy!(data(sz), datz)

    eqx = inject(sx, field=forward(fx), expr=sx)
    eqy = inject(sy, field=forward(fy), expr=sy)
    eqz = inject(sz, field=forward(fz), expr=sz)
    op = Operator([eqx, eqy, eqz], name="CoordOp")
    apply(op)

    x = convert(Array, data(fx))
    y = convert(Array, data(fy))
    z = convert(Array, data(fz))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if VERSION >= v"1.7"
            @test x ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0 1.0;;; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 2.0;;; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0; 3.0 3.0 3.0 3.0 3.0;;; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0; 4.0 4.0 4.0 4.0 4.0;;; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0; 5.0 5.0 5.0 5.0 5.0;;; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0; 6.0 6.0 6.0 6.0 6.0]
            @test y ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0;;; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0; 1.0 2.0 3.0 4.0 5.0]
            @test z ≈ [0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0;;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0;;; 1.0 1.0 1.0 1.0 1.0; 2.0 2.0 2.0 2.0 2.0; 3.0 3.0 3.0 3.0 3.0; 4.0 4.0 4.0 4.0 4.0]
        end
    end
end

@testset "Sparse time function, copy!, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    nt = 100
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=nt, grid=grid)

    x = Matrix{Float32}(undef,0,0)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x = reshape(Float32[1:prod(nt*npoint);], npoint, nt)
    end

    _x = data(stf)

    copy!(_x, x)
    x = reshape(Float32[1:prod(nt*npoint);], npoint, nt)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if npoint == 1
            @test isempty(parent(_x))
        elseif npoint == 5
            @test parent(_x) ≈ x[1:1,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[1:2,:]
        end
    elseif MPI.Comm_rank(MPI.COMM_WORLD) == 1
        if npoint == 1
            @test isempty(parent(_x))
        elseif npoint == 5
            @test parent(_x) ≈ x[2:2,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[3:4,:]
        end
    elseif MPI.Comm_rank(MPI.COMM_WORLD) == 2
        if npoint == 1
            @test isempty(parent(_x))
        elseif npoint == 5
            @test parent(_x) ≈ x[3:3,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[5:6,:]
        end
    elseif MPI.Comm_rank(MPI.COMM_WORLD) == 3
        if npoint == 1
            @test parent(_x) ≈ x
        elseif npoint == 5
            @test parent(_x) ≈ x[4:5,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[7:10,:]
        end
    end
end

@testset "Sparse time function, copy! and convert, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    nt = 100
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=nt, grid=grid)

    x = zeros(Float32, npoint, nt)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x .= reshape(Float32[1:prod(nt*npoint);], npoint, nt)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    _x = data(stf)
    @test isa(data(stf), Devito.DevitoMPISparseTimeArray)

    copy!(_x, x)
    x .= reshape(Float32[1:prod(nt*npoint);], npoint, nt)

    __x = convert(Array, _x)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test __x ≈ x
    end
end
