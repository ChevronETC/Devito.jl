
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
            @test isempty(parent(parent(_x)))
        elseif npoint == 5
            @test parent(parent(_x)) ≈ x[1:2,:]
        elseif npoint == 10
            @test parent(parent(_x)) ≈ x[1:5,:]
        end
    else
        if npoint == 1
            @test parent(parent(_x)) ≈ x
        elseif npoint == 5
            @test parent(parent(_x)) ≈ x[3:5,:]
        elseif npoint == 10
            @test parent(parent(_x)) ≈ x[6:10,:]
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
