
using Devito, LinearAlgebra, MPI, Random, Strided, Test

MPI.Init()
configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", true)

@testset "DevitoMPIArray, fill!, with halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})
    if length(n) == 2
        @test size(b_data) == (15,14)
    else
        @test size(b_data) == (16,15,14)
    end
    b_data .= 3.14f0

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if length(n) == 2
                @test parent(b_data) ≈ 3.14*ones(Float32, 15, 7)
            else
                @test parent(b_data) ≈ 3.14*ones(Float32, 16, 15, 7)
            end
            @test isa(parent(b_data), StridedView)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, fill!, no halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})
    @test size(b_data) == n
    b_data .= 3.14f0

    b_data_test = data(b)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if length(n) == 2
                @test parent(b_data_test) ≈ 3.14*ones(Float32, 11, 5)
            else
                @test parent(b_data_test) ≈ 3.14*ones(Float32, 12, 11, 5)
            end
            @test isa(parent(b_data_test), StridedView)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, copy!, inhalo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_inhalo(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})

    _n = length(n) == 2 ? (15,18) : (16,15,18)
    @show n, _n

    @test size(b_data) == _n

    b_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test = reshape(Float32[1:prod(_n);], _n)
    end
    copy!(b_data, b_data_test)
    b_data_test = reshape(Float32[1:prod(_n);], _n)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                if length(n) == 2
                    @test parent(b_data) ≈ b_data_test[:,1:9]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,1:9]
                end
            elseif rnk == 1
                if length(n) == 2
                    @test parent(b_data) ≈ b_data_test[:,10:18]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,10:18]
                end
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, copy!, halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})

    _n = length(n) == 2 ? (15,14) : (16,15,14)

    @test size(b_data) == _n

    b_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test = reshape(Float32[1:prod(_n);], _n)
    end
    copy!(b_data, b_data_test)
    b_data_test = reshape(Float32[1:prod(_n);], _n)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                if length(n) == 2
                    @test parent(b_data) ≈ b_data_test[:,1:7]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,1:7]
                end
            end
            if rnk == 1
                if length(n) == 2
                    @test parent(b_data) ≈ b_data_test[:,8:14]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,8:14]
                end
            end
            @test isa(parent(b_data), StridedView)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, copy!, no halo, n=$n" for n in ( (11,10), (12,11,10) )
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

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[:,1:5]
                    @test parent(b_data) ≈ b_data_test[:,1:5]
                else
                    @test parent(_b_data) ≈ b_data_test[:,:,1:5]
                    @test parent(b_data) ≈ b_data_test[:,:,1:5]
                end
                @test isa(parent(_b_data), StridedView)
            end
            if rnk == 1
                if length(n) == 2
                    @test parent(_b_data) ≈ b_data_test[:,6:10]
                    @test parent(b_data) ≈ b_data_test[:,6:10]
                else
                    @test parent(_b_data) ≈ b_data_test[:,:,6:10]
                    @test parent(b_data) ≈ b_data_test[:,:,6:10]
                end
                @test isa(parent(_b_data), StridedView)
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "convert data from rank 0 to DevitoMPIArray, then back, inhalo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_inhalo(b)

    _n = length(n) == 2 ? (15,18) : (16,15,18)

    b_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    copy!(b_data, b_data_test)

    b_data_out = convert(Array, b_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_out ≈ b_data_test
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

@testset "convert data from rank 0 to DevitoMPIArray, then back, halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)

    _n = length(n) == 2 ? (15,14) : (16,15,14)

    b_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(b_data, b_data_test)
    _b_data = data_with_halo(b)

    b_data_out = convert(Array, _b_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_out ≈ b_data_test
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

@testset "DevitoMPITimeArray, copy!, data, inhalo, n=$n" for n in ( (11,10), (12,11,10))
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_inhalo(p)

    _n = length(n) == 2 ? (15,18,3) : (16,15,18,3)
    p_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        p_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(p_data, p_data_test)
    p_data_test .= reshape(Float32[1:prod(_n);], _n)

    p_data_local = parent(p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if length(n) == 2
            @test p_data_local ≈ p_data_test[:,1:9,:]
        else
            @test p_data_local ≈ p_data_test[:,:,1:9,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 1
        if length(n) == 2
            @test p_data_local ≈ p_data_test[:,10:18,:]
        else
            @test p_data_local ≈ p_data_test[:,:,10:18,:]
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

@testset "DevitoMPITimeArray, copy!, data, halo, n=$n" for n in ( (11,10), (12,11,10))
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_halo(p)

    _n = length(n) == 2 ? (15,14,3) : (16,15,14,3)
    p_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        p_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(p_data, p_data_test)
    p_data_test .= reshape(Float32[1:prod(_n);], _n)

    p_data_local = parent(p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if length(n) == 2
            @test p_data_local ≈ p_data_test[:,1:7,:]
        else
            @test p_data_local ≈ p_data_test[:,:,1:7,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 1
        if length(n) == 2
            @test p_data_local ≈ p_data_test[:,8:14,:]
        else
            @test p_data_local ≈ p_data_test[:,:,8:14,:]
        end
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
            @test p_data_local ≈ p_data_test[:,1:5,:]
        else
            @test p_data_local ≈ p_data_test[:,:,1:5,:]
        end
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 1
        if length(n) == 2
            @test p_data_local ≈ p_data_test[:,6:10,:]
        else
            @test p_data_local ≈ p_data_test[:,:,6:10,:]
        end
    end
    MPI.Barrier(MPI.COMM_WORLD)
end

@testset "convert data from rank 0 to DevitoMPITimeArray, then back, inhalo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_inhalo(p)

    _n = length(n) == 2 ? (15,18,3) : (16,15,18,3)

    p_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        p_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(p_data, p_data_test)
    p_data_test .= reshape(Float32[1:prod(_n);], _n)

    _p_data_test = convert(Array, p_data)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test p_data_test ≈ _p_data_test
    end
end

@testset "convert data from rank 0 to DevitoMPITimeArray, then back, halo, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_halo(p)

    _n = length(n) == 2 ? (15,14,3) : (16,15,14,3)

    p_data_test = zeros(Float32, _n)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        p_data_test .= reshape(Float32[1:prod(_n);], _n)
    end
    copy!(p_data, p_data_test)
    p_data_test .= reshape(Float32[1:prod(_n);], _n)

    _p_data_test = convert(Array, p_data)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test p_data_test ≈ _p_data_test
    end
end

@testset "convert data from rank 0 to DevitoMPITimeArray, then back, no halo, n=$n" for n in ( (11,10), (12,11,10) )
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

    _p_data_test = convert(Array, p_data)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test p_data_test ≈ _p_data_test
    end
end

@testset "DevitoMPITimeArray coordinates check, 2D" begin
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

@testset "Sparse time function coordinates, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=100, grid=grid)
    stf_coords = coordinates(stf)
    @test isa(stf_coords, Devito.DevitoMPIArray)
    @test size(stf_coords) == (length(n),npoint)

    x = reshape(Float32[1:length(n)*npoint;], length(n), npoint)

    copy!(stf_coords, x)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if npoint == 1
            @test isempty(parent(stf_coords))
        elseif npoint == 5
            @test parent(stf_coords) ≈ (x[:,1:2])
        elseif npoint == 10
            @test parent(stf_coords) ≈ (x[:,1:5])
        end
    else
        if npoint == 1
            @test parent(stf_coords) ≈ x
        elseif npoint == 5
            @test parent(stf_coords) ≈ (x[:,3:end])
        elseif npoint == 10
            @test parent(stf_coords) ≈ (x[:,6:end])
        end
    end

    # round trip
    _stf_coords = convert(Array,coordinates(stf))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test _stf_coords ≈ x
    end
end

@testset "Sparse time function size npoint=$npoint" for npoint in (1,5) 
    grid = Grid(shape=(11,12), dtype=Float32)
    nt = 100
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=nt, grid=grid)
    @test size(stf) == (npoint,nt)
    @test size_with_halo(stf) == (npoint,nt)
end

@testset "Sparse time function dispatch" begin
    grid = Grid(shape=(11,12), dtype=Float32)
    nt = 100
    stf = SparseTimeFunction(name="stf", npoint=1, nt=nt, grid=grid)
    x = ones(Float64, 1, nt)
    data_stf = data(stf)
    @test_throws ErrorException copy!(data_stf,x) 
    x = ones(Float32, 1, nt)
    copy!(data_stf, x)
    if MPI.Comm_rank == 0
        @test convert(Array,data_stf) ≈ x
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
            @test parent(_x) ≈ x[1:2,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[1:5,:]
        end
    else
        if npoint == 1
            @test parent(_x) ≈ x
        elseif npoint == 5
            @test parent(_x) ≈ x[3:5,:]
        elseif npoint == 10
            @test parent(_x) ≈ x[6:10,:]
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

@testset "DevitoMPISparseArray copy! axes check, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
    stf_data = data(stf)
    @test size(stf_data) == (10,100)
    x = rand(100,10)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test_throws ArgumentError copy!(stf_data, x)
    end
end

@testset "MPI Setindex Not Implemented" begin
    grid = Grid(shape=(5,6,7))
    f = Devito.Function(name="f", grid=grid)
    @test_throws ErrorException("not implemented") data(f)[2,2,2] = 1.0
end

@testset "MPI Getindex" for n in ( (11,10), (5,4), (7,2), (4,5,6), (2,3,4) )
    N = length(n)
    rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=Float32)
    f = Devito.Function(name="f", grid=grid)
    arr = reshape(1f0*[1:prod(size(grid));], size(grid))
    copy!(data(f), arr)
    nchecks = 10
    Random.seed!(1234);
    for check in 1:nchecks
        i = rand((1:n[1]))
        j = rand((1:n[2]))
        I = (i,j)
        if N == 3
            k = rand((1:n[3]))
            I = (i,j,k)
        end
       @test data(f)[I...] == arr[I...]
    end
    if N == 2
        @test data(f)[1:div(n[1],2),:] ≈ arr[1:div(n[1],2),:]
    else
        @test data(f)[1:div(n[1],2),div(n[2],3):2*div(n[2],3),:] ≈ arr[1:div(n[1],2),div(n[2],3):2*div(n[2],3),:]
    end
end
