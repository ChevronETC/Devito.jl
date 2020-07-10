using Devito, Random, PyCall, Test

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", false)

@testset "Grid" begin
    grid = Grid(shape = (4,5), extent=(40.0,50.0), dtype = Float32)
    @test size(grid) == (5,4)
    @test ndims(grid) == 2
    @test eltype(grid) == Float32
end

@testset "Function, data_with_halo" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)

    rand!(b_data)

    b_data_test = data_with_halo(b)
    @test b_data ≈ b_data_test
end

@testset "Function, data" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)

    copy!(b_data, rand(eltype(grid), size(grid)))

    b_data_test = data(b)
    @test b_data ≈ b_data_test
end

@testset "TimeFunction, data with halo" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_halo(p)

    copy!(p_data, rand(eltype(grid), size_with_halo(p)))

    p_data_test = data_with_halo(p)
    @test p_data ≈ p_data_test
end

@testset "TimeFunction, data" begin
    grid = Grid(shape = (4,5), dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data(p)

    copy!(p_data, rand(eltype(grid), size(p)))

    p_data_test = data(p)
    @test p_data ≈ p_data_test
end

using Distributed, MPIClusterManagers

manager = MPIManager(;np=2)
addprocs(manager)

@everywhere using Devito, MPI, Random, Test

@mpi_do manager begin
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
end

rmprocs(workers())