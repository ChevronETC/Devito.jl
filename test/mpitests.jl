using Distributed, MPIClusterManagers

manager = MPIManager(;np=2)
addprocs(manager)

@everywhere using Devito, LinearAlgebra, MPI, Random, Strided, Test

@mpi_do manager begin
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

    @testset "DevitoMPIArray, copy!, halo, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape=n, dtype=Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        b_data = data_with_halo(b)
        @test isa(b_data, Devito.DevitoMPIArray{Float32,length(n)})

        _n = length(n) == 2 ? (15,14) : (16,15,14)

        @test size(b_data) == _n

        b_data_test = rand(Float32,_n)
        copy!(b_data, b_data_test)

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
        b_data_test = reshape([1:prod(n);], n)
        copy!(b_data, b_data_test)
        _b_data = data(b)

        for rnk in 0:1
            if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
                if rnk == 0
                    if length(n) == 2
                        @test parent(_b_data) ≈ b_data_test[:,1:5]
                    else
                        @test parent(_b_data) ≈ b_data_test[:,:,1:5]
                    end
                    @test isa(parent(_b_data), StridedView)
                end
                if rnk == 1
                    if length(n) == 2
                        @test parent(_b_data) ≈ b_data_test[:,6:10]
                    else
                        @test parent(_b_data) ≈ b_data_test[:,:,6:10]
                    end
                    @test isa(parent(_b_data), StridedView)
                end
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end

    @testset "Convert data from rank 0 to DevitoMPIArray then back n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape=n, dtype=Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        b_data = data(b)

        b_data_test = zeros(n)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            b_data_test .= reshape([1:prod(n);], n)
        end
        copy!(b_data, b_data_test)
        _b_data = data(b)

        b_data_out = convert(Array,_b_data)
        MPI.Barrier(MPI.COMM_WORLD)
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test b_data_out ≈ b_data_test
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end

    @testset "DevitoMPIArray, convert to Array, halo, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape=n, dtype=Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        b_data = data_with_halo(b)

        _n = length(n) == 2 ? (15,14) : (16,15,14)

        b_data_test = reshape([1:prod(_n);], _n)
        copy!(b_data, b_data_test)

        _b_data_test = convert(Array, b_data)
        
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test b_data_test ≈ _b_data_test
        end
    end

    @testset "DevitoMPIArray, convert to Array, no halo, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape=n, dtype=Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        b_data = data(b)

        b_data_test = reshape([1:prod(n);], n)
        copy!(b_data, b_data_test)

        _b_data_test = convert(Array, b_data)
        
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test b_data_test ≈ _b_data_test
        end
    end

    @testset "TimeFunction, data with halo, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape = n, dtype = Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
        p_data = data_with_halo(p)

        _n = length(n) == 2 ? (15,14,3) : (16,15,14,3)

        @show size(p_data)
        @show p_data.local_indices
        p_data_test = reshape([1:prod(_n);], _n)
        copy!(p_data, p_data_test)

        _p_data_test = convert(Array, p_data)

        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test p_data_test ≈ _p_data_test
        end
    end

    @testset "TimeFunction, data with no halo, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape = n, dtype = Float32)
        b = Devito.Function(name="b", grid=grid, space_order=2)
        p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
        p_data = data(p)

        _n = length(n) == 2 ? (11,10,3) : (12,11,10,3)

        p_data_test = reshape([1:prod(_n);], _n)
        copy!(p_data, p_data_test)

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
        x = rand(length(n),npoint)
        copy!(stf_coords, x)
        _stf_coords = convert(Array,coordinates(stf))
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test _stf_coords ≈ x
        end
    end

    @testset "DevitoMPISparseArray, copy! and convert, transposed" begin
        grid = Grid(shape=(11,10), dtype=Float32)
        stf = SparseTimeFunction(name="stf", grid=grid, npoint=1, nt=100)
        x = rand(Float32,100,1)
        copy!(data(stf), x)
        @test isa(data(stf), Transpose)
        _x = convert(Array, data(stf))
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            @test x ≈ _x
        end
    end

    @testset "DevitoMPISparseArray copy! axes check, n=$n" for n in ( (11,10), (12,11,10) )
        grid = Grid(shape=n, dtype=Float32)
        stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
        stf_data = data(stf)
        @test size(stf_data) == (100,10)
        x = rand(10,100)
        @test_throws ArgumentError copy!(stf_data, x)
    end
        
end

rmprocs(workers())
