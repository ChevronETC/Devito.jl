using Devito, MPI, Random, Test

MPI.Init()

if MPI.Comm_size(MPI.COMM_WORLD) != 2
    error("expecting MPI comm size of 2")
end

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", true)

@testset "DevitoMPIArray, fill!, with halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,2})
    @test size(b_data) == (9,8)
    b_data .= 3.14f0

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            @test parent(b_data) ≈ 3.14*ones(Float32, 9, 4)
            @test isa(parent(b_data), SubArray)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, fill!, no halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,2})
    @test size(b_data) == (5,4)
    b_data .= 3.14f0

    b_data_test = data(b)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            @test parent(b_data_test) ≈ 3.14*ones(Float32, 5, 2)
            @test isa(parent(b_data_test), SubArray)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, copy!, halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,2})
    @test size(b_data) == (9,8)

    b_data_test = rand(Float32,9,8)
    copy!(b_data, b_data_test)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                @test parent(b_data) ≈ b_data_test[:,1:4]
            end
            if rnk == 1
                @test parent(b_data) ≈ b_data_test[:,5:8]
            end
            @test isa(parent(b_data), SubArray)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, copy!, no halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)
    @test isa(b_data, Devito.DevitoMPIArray{Float32,2})
    @test size(b_data) == (5,4)

    b_data_test = reshape([1:20;], (5,4))
    copy!(b_data, b_data_test)
    _b_data = data(b)

    for rnk in 0:1
        if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
            if rnk == 0
                @test parent(_b_data) ≈ b_data_test[:,1:2]
                @test isa(parent(_b_data), SubArray)
            end
            if rnk == 1
                @test parent(_b_data) ≈ b_data_test[:,3:4]
                @test isa(parent(_b_data), SubArray)
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end

@testset "DevitoMPIArray, convert to Array, halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)

    b_data_test = reshape([1:72;], (9,8))
    copy!(b_data, b_data_test)

    _b_data_test = convert(Array, b_data)
    
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_test ≈ _b_data_test
    end
end

@testset "DevitoMPIArray, convert to Array, no halo" begin
    grid = Grid(shape=(4,5), dtype=Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)

    b_data_test = reshape([1:20;], (5,4))
    copy!(b_data, b_data_test)

    _b_data_test = convert(Array, b_data)
    
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_test ≈ _b_data_test
    end
end

@testset "TimeFunction, data with halo" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_halo(p)

    @show size(p_data)
    @show p_data.local_indices
    p_data_test = reshape([1:216;], (9,8,3))
    copy!(p_data, p_data_test)

    _p_data_test = convert(Array, p_data)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test p_data_test ≈ _p_data_test
    end
end

@testset "TimeFunction, data with no halo" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data(p)

    p_data_test = reshape([1:60;], (5,4,3))
    copy!(p_data, p_data_test)

    _p_data_test = convert(Array, p_data)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test p_data_test ≈ _p_data_test
    end
end

MPI.Finalize()