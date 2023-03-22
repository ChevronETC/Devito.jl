
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

@testset "DevitoMPIArray localsize, n=$n" for n in ((5,4),(6,5,4))
    g = Grid(shape=n)
    f = Devito.Function(name="f", grid=g)
    h = Devito.TimeFunction(name="h", grid=g, time_order=2)
    for func in (f,h)
        @test localsize(data(func)) == length.(Devito.localindices(data(func)))
    end
end

@testset "DevitoMPISparseArray localsize, n=$n, npoint=$npoint" for n in ((5,4),(6,5,4)), npoint in (1,5,10)
    g = Grid(shape=n)
    sf = SparseFunction(name="sf", grid=g, npoint=npoint)
    @test localsize(data(sf)) == length.(Devito.localindices(data(sf)))
end

@testset "DevitoMPISparseTimeArray localsize, n=$n, npoint=$npoint" for n in ((5,4),(6,5,4)), npoint in (1,5,10)
    g = Grid(shape=n)
    nt = 11
    stf = SparseTimeFunction(name="stf", grid=g, nt=11, npoint=npoint)
    @test localsize(data(stf)) == length.(Devito.localindices(data(stf)))
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

@testset "convert data from rank 0 to DevitoSparseMPIArray, then back, extra dimension n=$n, nextra=$nextra, npoint=$npoint, first=$first" for n in ( (11,10), (12,11,10) ), nextra in (1,2,5), first in (true,false), npoint in (1,2,5)
    grid = Grid(shape = n, dtype = Float32)
    extradim = Dimension(name="extra")
    space_order = 2
    prec = Dimension(name="prec")
    npoint = 8
    sparsedims = (extradim, prec)
    sparseshape = (nextra, npoint)
    if ~first
        sparsedims = reverse(sparsedims)
        sparseshape = reverse(sparseshape)
        sf = SparseFunction(name="sf", grid=grid, dimensions=sparsedims, shape=sparseshape, npoint=npoint)
    else
        sf = CoordSlowSparseFunction(name="sf", grid=grid, dimensions=sparsedims, shape=sparseshape, npoint=npoint)
    end

    sf_data_test = zeros(Float32, sparseshape...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        sf_data_test .= reshape(Float32[1:prod(sparseshape);], sparseshape)
    end
    copy!( data(sf), sf_data_test)
    _sf_data_test = convert(Array, data(sf))
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test sf_data_test ≈ _sf_data_test
    end
end

@testset "convert data from rank 0 to DevitoSparseMPITimeArray, then back, extra dimension n=$n, nextra=$nextra, npoint=$npoint, nt=$nt" for n in ( (11,10), (12,11,10) ), nextra in (1,2,5), npoint in (1,2,5), nt in (1,5,10)
    grid = Grid(shape = n, dtype = Float32)
    extradim = Dimension(name="extra")
    space_order = 2
    t = time_dim(grid)
    prec = Dimension(name="prec")
    npoint = 8
    sparsedims = (prec, extradim, t)
    sparseshape = (npoint, nextra, nt)
    sf = SparseTimeFunction(name="sf", grid=grid, dimensions=sparsedims, shape=sparseshape, npoint=npoint, nt=nt)

    sf_data_test = zeros(Float32, sparseshape...)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        sf_data_test .= reshape(Float32[1:prod(sparseshape);], sparseshape)
    end
    copy!( data(sf), sf_data_test)
    _sf_data_test = convert(Array, data(sf))
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test sf_data_test ≈ _sf_data_test
    end

end

@testset "convert data from rank 0 to DevitoMPIArray, then back, halo, n=$n" for n in ( (11,10), (12,11,10) )
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

@testset "convert data from rank 0 to DevitoMPIArray, then back, extra dimension, n=$n, nextra=$nextra, first=$first" for n in ( (11,10), (12,11,10) ), nextra in (1,2,5), first in (true,false)
    grid = Grid(shape = n, dtype = Float32)
    extradim = Dimension(name="extra")
    space_order = 2
    time = stepping_dim(grid)
    if first 
        funcdims = (extradim, dimensions(grid)...)
        funcshap = (nextra, n...)
    else
        funcdims = (dimensions(grid)..., extradim)
        funcshap = (n..., nextra)
    end
    timedims = (funcdims..., time)
    timeshap = (funcshap..., 2)

    b = Devito.Function(name="b", grid=grid, space_order=space_order, dimensions=funcdims, shape=funcshap)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=space_order, dimensions=timedims, shape=timeshap)
    b_data = data(b)
    p_data = data(p)

    b_data_test = zeros(Float32, funcshap)
    p_data_test = zeros(Float32, timeshap)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(funcshap);], funcshap)
        p_data_test .= reshape(Float32[1:prod(timeshap);], timeshap)
    end
    copy!(b_data, b_data_test)
    copy!(p_data, p_data_test)
    b_data_test .= reshape(Float32[1:prod(funcshap);], funcshap)
    p_data_test .= reshape(Float32[1:prod(timeshap);], timeshap)
    _b_data_test = convert(Array, b_data)
    _p_data_test = convert(Array, p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_test ≈ _b_data_test
        @test p_data_test ≈ _p_data_test
    end
end

@testset "convert data from rank 0 to DevitoMPIArray, then back, extra dimensions inhalo, n=$n, nextra=$nextra, first=$first" for n in ( (11,10), (12,11,10) ), nextra in (1,2,5), first in (false,true)
    grid = Grid(shape = n, dtype = Float32)
    extradim = Dimension(name="extra")
    space_order = 2
    time = stepping_dim(grid)
    padded = (length(n) == 2 ? (15, 18) : (16, 15, 18))
    if first 
        funcdims = (extradim, dimensions(grid)...)
        funcshap = (nextra, n...)
        arrayshap = (nextra, padded...)
    else
        funcdims = (dimensions(grid)..., extradim)
        funcshap = (n..., nextra)
        arrayshap = (padded..., nextra)
    end
    timedims = (funcdims..., time)
    timeshap = (funcshap..., 2)
    timearrayshap = (arrayshap..., 2)

    b = Devito.Function(name="b", grid=grid, space_order=space_order, dimensions=funcdims, shape=funcshap)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=space_order, dimensions=timedims, shape=timeshap)
    b_data = data_with_inhalo(b)
    p_data = data_with_inhalo(p)
    b_data_test = zeros(Float32, arrayshap)
    p_data_test = zeros(Float32, timearrayshap)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(arrayshap);], arrayshap)
        p_data_test .= reshape(Float32[1:prod(timearrayshap);], timearrayshap)
    end
    copy!(b_data, b_data_test)
    copy!(p_data, p_data_test)
    b_data_test .= reshape(Float32[1:prod(arrayshap);], arrayshap)
    p_data_test .= reshape(Float32[1:prod(timearrayshap);], timearrayshap)
    _b_data_test = convert(Array, b_data)
    _p_data_test = convert(Array, p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_test ≈ _b_data_test
        @test p_data_test ≈ _p_data_test
    end
end

@testset "convert data from rank 0 to DevitoMPIArray, then back, extra dimensions with halo, n=$n, nextra=$nextra, first=$first" for n in ( (11,10), (12,11,10) ), nextra in (1,2,5), first in (false,true)
    grid = Grid(shape = n, dtype = Float32)
    extradim = Dimension(name="extra")
    space_order = 2
    time = stepping_dim(grid)
    padded = (length(n) == 2 ? (15, 14) : (16, 15, 14))
    if first 
        funcdims = (extradim, dimensions(grid)...)
        funcshap = (nextra, n...)
        arrayshap = (nextra, padded...)
    else
        funcdims = (dimensions(grid)..., extradim)
        funcshap = (n..., nextra)
        arrayshap = (padded..., nextra)
    end
    timedims = (funcdims..., time)
    timeshap = (funcshap..., 2)
    timearrayshap = (arrayshap..., 2)
    
    b = Devito.Function(name="b", grid=grid, space_order=space_order, dimensions=funcdims, shape=funcshap)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=space_order, dimensions=timedims, shape=timeshap)
    b_data = data_with_halo(b)
    p_data = data_with_halo(p)
    b_data_test = zeros(Float32, arrayshap)
    p_data_test = zeros(Float32, timearrayshap)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        b_data_test .= reshape(Float32[1:prod(arrayshap);], arrayshap)
        p_data_test .= reshape(Float32[1:prod(timearrayshap);], timearrayshap)
    end
    copy!(b_data, b_data_test)
    copy!(p_data, p_data_test)
    b_data_test .= reshape(Float32[1:prod(arrayshap);], arrayshap)
    p_data_test .= reshape(Float32[1:prod(timearrayshap);], timearrayshap)
    _b_data_test = convert(Array, b_data)
    _p_data_test = convert(Array, p_data)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test b_data_test ≈ _b_data_test
        @test p_data_test ≈ _p_data_test
    end
end

@testset "DevitoMPITimeArray coordinates check, 2D" begin
    ny,nx = 4,6

    grd = Grid(shape=(ny,nx), extent=(ny-1,nx-1), dtype=Float32)
    time_order = 1
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, save=time_order+1, allowpro=false)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, save=time_order+1, allowpro=false)
    
    sx = SparseTimeFunction(name="sx", grid=grd, npoint=ny*nx, nt=time_order+1)
    sy = SparseTimeFunction(name="sy", grid=grd, npoint=ny*nx, nt=time_order+1)

    cx = [ix-1 for iy = 1:ny, ix=1:nx][:]
    cy = [iy-1 for iy = 1:ny, ix=1:nx][:]

    coords = zeros(Float32, 2, ny*nx)
    coords[1,:] .= cx
    coords[2,:] .= cy

    copy!(coordinates_data(sx), coords)
    copy!(coordinates_data(sy), coords)

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
    fx = TimeFunction(name="fx", grid=grd, time_order=time_order, allowpro=false, save=time_order+1)
    fy = TimeFunction(name="fy", grid=grd, time_order=time_order, allowpro=false, save=time_order+1)
    fz = TimeFunction(name="fz", grid=grd, time_order=time_order, allowpro=false, save=time_order+1)
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
    copy!(coordinates_data(sx), coords)
    copy!(coordinates_data(sy), coords)
    copy!(coordinates_data(sz), coords)

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

@testset "Sparse function coordinates, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    sf = SparseFunction(name="sf", npoint=npoint, grid=grid)
    sf_coords = coordinates_data(sf)
    @test isa(sf_coords, Devito.DevitoMPIArray)
    @test size(sf_coords) == (length(n),npoint)

    x = reshape(Float32[1:length(n)*npoint;], length(n), npoint)

    copy!(sf_coords, x)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if npoint == 1
            @test isempty(parent(sf_coords))
        elseif npoint == 5
            @test parent(sf_coords) ≈ (x[:,1:2])
        elseif npoint == 10
            @test parent(sf_coords) ≈ (x[:,1:5])
        end
    else
        if npoint == 1
            @test parent(sf_coords) ≈ x
        elseif npoint == 5
            @test parent(sf_coords) ≈ (x[:,3:end])
        elseif npoint == 10
            @test parent(sf_coords) ≈ (x[:,6:end])
        end
    end

    # round trip
    _sf_coords = convert(Array,coordinates_data(sf))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test _sf_coords ≈ x
    end
end

@testset "Sparse time function coordinates, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=100, grid=grid)
    stf_coords = coordinates_data(stf)
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
    _stf_coords = convert(Array,coordinates_data(stf))

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test _stf_coords ≈ x
    end
end

@testset "Sparse function size npoint=$npoint" for npoint in (1,5) 
    grid = Grid(shape=(11,12), dtype=Float32)
    nt = 100
    sf = SparseFunction(name="sf", npoint=npoint, grid=grid)
    @test size(sf) == (npoint,)
    @test size_with_halo(sf) == (npoint,)
end

@testset "Sparse time function size npoint=$npoint" for npoint in (1,5) 
    grid = Grid(shape=(11,12), dtype=Float32)
    nt = 100
    stf = SparseTimeFunction(name="stf", npoint=npoint, nt=nt, grid=grid)
    @test size(stf) == (npoint,nt)
    @test size_with_halo(stf) == (npoint,nt)
end

@testset "Sparse function, copy!, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    sf = SparseFunction(name="sf", npoint=npoint, grid=grid)

    x = Array{Float32}(undef,0)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x = Float32[1:npoint;]
    end
    
    _x = data(sf)
    copy!(_x, x)
    
    x = Float32[1:npoint;]
    
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        if npoint == 1
            @test isempty(parent(_x))
        elseif npoint == 5
            @test parent(_x) ≈ x[1:2]
        elseif npoint == 10
            @test parent(_x) ≈ x[1:5]
        end
    else
        if npoint == 1
            @test parent(_x) ≈ x
        elseif npoint == 5
            @test parent(_x) ≈ x[3:5]
        elseif npoint == 10
            @test parent(_x) ≈ x[6:10]
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

@testset "Sparse function, copy! and convert, n=$n, npoint=$npoint" for n in ( (11,10), (12,11,10) ), npoint in (1, 5, 10)
    grid = Grid(shape=n, dtype=Float32)
    sf = SparseFunction(name="sf", npoint=npoint, grid=grid)

    x = zeros(Float32, npoint)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        x .= Float32[1:npoint;]
    end
    MPI.Barrier(MPI.COMM_WORLD)
    _x = data(sf)
    @test isa(data(sf), Devito.DevitoMPISparseArray)

    copy!(_x, x)
    x .= Float32[1:npoint;]

    __x = convert(Array, _x)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test __x ≈ x
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

@testset "DevitoMPISparseTimeArray copy! axes check, n=$n" for n in ( (11,10), (12,11,10) )
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
    stf_data = data(stf)
    @test size(stf_data) == (10,100)
    x = rand(100,10)
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @test_throws ArgumentError copy!(stf_data, x)
    end
end

@testset "MPI Getindex for Function n=$n" for n in ( (11,10), (5,4), (7,2), (4,5,6), (2,3,4) )
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

@testset "MPI Getindex for TimeFunction n=$n" for n in ( (11,10), (5,4), (7,2), (4,5,6), (2,3,4) )
    N = length(n)
    nt = 5
    rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=Float32)
    f = TimeFunction(name="f", grid=grid, save=nt, allowpro=false)
    arr = reshape(1f0*[1:prod(size(data(f)));], size(data(f)))
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
        m = rand((1:nt))
        I = (I...,m)
        @test data(f)[I...] == arr[I...]
    end
    if N == 2
        @test data(f)[1:div(n[1],2),:,1:div(nt,2)] ≈ arr[1:div(n[1],2),:,1:div(nt,2)]
    else
        @test data(f)[1:div(n[1],2),div(n[2],3):2*div(n[2],3),:,1:div(nt,2)] ≈ arr[1:div(n[1],2),div(n[2],3):2*div(n[2],3),:,1:div(nt,2)]
    end
end

@testset "MPI Getindex for SparseFunction n=$n npoint=$npoint" for n in ( (5,4),(4,5,6) ), npoint in (1,5,10)
    N = length(n)
    nt = 5
    rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=Float32)
    f = SparseFunction(name="f", grid=grid, npoint=npoint)
    arr = reshape(1f0*[1:prod(size(data(f)));], size(data(f)))
    copy!(data(f), arr)
    nchecks = 10
    Random.seed!(1234);
    for check in 1:nchecks
        i = rand((1:npoint))
        I = (i,)
        @test data(f)[I...] == arr[I...]
    end
    if npoint > 1
        @test data(f)[1:div(npoint,2)] ≈ arr[1:div(npoint,2)]
    else
        @test data(f)[1] == arr[1]
    end
end

@testset "MPI Getindex for SparseTimeFunction n=$n npoint=$npoint" for n in ( (5,4),(4,5,6) ), npoint in (1,5,10)
    N = length(n)
    nt = 5
    rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=Float32)
    f = SparseTimeFunction(name="f", grid=grid, nt=nt, npoint=npoint)
    arr = reshape(1f0*[1:prod(size(data(f)));], size(data(f)))
    copy!(data(f), arr)
    nchecks = 10
    Random.seed!(1234);
    for check in 1:nchecks
        i = rand((1:npoint))
        j = rand((1:nt))
        I = (i,j)
        @test data(f)[I...] == arr[I...]
    end
    if npoint > 1
        @test data(f)[1:div(npoint,2),2:end-1] ≈ arr[1:div(npoint,2),2:end-1]
    else
        @test data(f)[1,2:end-1] ≈ arr[1,2:end-1]
    end
end

@testset "MPI setindex! for Function n=$n, T=$T" for n in ( (11,10), (5,4), (7,2), (4,5,6), (2,3,4) ), T in (Float32,Float64)
    N = length(n)
    my_rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=T)
    f = Devito.Function(name="f", grid=grid)
    base_array = reshape(one(T)*[1:prod(size(grid));], size(grid))

    send_arr = zeros(T, (0 .* n)...)
    expected_arr = zeros(T, (0 .* n)...)

    if my_rnk == 0
        send_arr = base_array
        expected_arr = zeros(T, n...)
    end

    nchecks = 10
    Random.seed!(1234);
    local indexes
    if N == 2
        indexes = [rand((1:n[1]),nchecks) rand((1:n[2]),nchecks) ;]
    else
        indexes = [rand((1:n[1]),nchecks) rand((1:n[2]),nchecks) rand((1:n[3]),nchecks);]
    end
    for check in 1:nchecks
        data(f)[indexes[check,:]...] = (my_rnk == 0 ? send_arr[indexes[check,:]...] : zero(T))
        if my_rnk == 0 
            expected_arr[indexes[check,:]...] = base_array[indexes[check,:]...]
        end
        @test data(f)[indexes[check,:]...] == base_array[indexes[check,:]...]
    end
    made_array = convert(Array,data(f))
    if my_rnk == 0
        @test made_array ≈ expected_arr
    end
end

@testset "MPI setindex! for TimeFunction n=$n, T=$T" for n in ( (11,10), (5,4), (7,2), (4,5,6), (2,3,4) ), T in (Float32,Float64)
    N = length(n)
    time_order = 2
    my_rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=T)
    f = TimeFunction(name="f", grid=grid, time_order=time_order)
    base_array = reshape(one(T)*[1:prod(size(data(f)));], size(data(f)))

    send_arr = zeros(T, (0 .* size(data(f)))...)
    expected_arr = zeros(T, (0 .* size(data(f)))...)

    if my_rnk == 0
        send_arr = base_array
        expected_arr = zeros(T, size(base_array)...)
    end

    nchecks = 10
    Random.seed!(1234);
    local indexes
    if N == 2
        indexes = [rand((1:n[1]),nchecks) rand((1:n[2]),nchecks) rand((1:time_order+1),nchecks);]
    else
        indexes = [rand((1:n[1]),nchecks) rand((1:n[2]),nchecks) rand((1:n[3]),nchecks) rand((1:time_order+1),nchecks);]
    end
    for check in 1:nchecks
        data(f)[indexes[check,:]...] = (my_rnk == 0 ? send_arr[indexes[check,:]...] : zero(T) )
        if my_rnk == 0 
            expected_arr[indexes[check,:]...] = base_array[indexes[check,:]...]
        end
        @test data(f)[indexes[check,:]...] == base_array[indexes[check,:]...]
    end
    made_array = convert(Array,data(f))
    if my_rnk == 0
        @test made_array ≈ expected_arr
    end
end

@testset "MPI settindex! for SparseTimeFunction n=$n, npoint=$npoint, T=$T" for n in ( (5,4),(4,5,6) ), npoint in (1,5,10), T in (Float32,Float64)
    N = length(n)
    nt = 11
    my_rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=T)
    f = SparseTimeFunction(name="f", grid=grid, nt=nt, npoint=npoint)
    base_array = reshape(one(T)*[1:prod(size(data(f)));], size(data(f)))
    send_arr = zeros(T, (0 .* size(data(f)))...)
    expected_arr = zeros(T, (0 .* size(data(f)))...)

    if my_rnk == 0
        send_arr = base_array
        expected_arr = zeros(T, size(base_array)...)
    end
    
    nchecks = 10
    Random.seed!(1234);
    indexes = [rand((1:npoint),nchecks) rand((1:nt),nchecks);]

    for check in 1:nchecks
        data(f)[indexes[check,:]...] = (my_rnk == 0 ? send_arr[indexes[check,:]...] : zero(T) )
        if my_rnk == 0 
            expected_arr[indexes[check,:]...] = base_array[indexes[check,:]...]
        end
        @test data(f)[indexes[check,:]...] == base_array[indexes[check,:]...]
    end
    made_array = convert(Array,data(f))
    if my_rnk == 0
        @test made_array ≈ expected_arr
    end
end

@testset "MPI settindex! for SparseFunction n=$n, npoint=$npoint, T=$T" for n in ( (5,4),(4,5,6) ), npoint in (1,5,10), T in (Float32,Float64)
    N = length(n)
    my_rnk = MPI.Comm_rank(MPI.COMM_WORLD)
    grid = Grid(shape=n, dtype=T)
    f = SparseFunction(name="f", grid=grid, npoint=npoint)
    base_array = reshape(one(T)*[1:prod(size(data(f)));], size(data(f)))
    send_arr = zeros(T, (0 .* size(data(f)))...)
    expected_arr = zeros(T, (0 .* size(data(f)))...)

    if my_rnk == 0
        send_arr = base_array
        expected_arr = zeros(T, size(base_array)...)
    end
    
    nchecks = 10
    Random.seed!(1234);
    indexes = [rand((1:npoint),nchecks);]

    for check in 1:nchecks
        data(f)[indexes[check,:]...] = (my_rnk == 0 ? send_arr[indexes[check,:]...] : zero(T) )
        if my_rnk == 0 
            expected_arr[indexes[check,:]...] = base_array[indexes[check,:]...]
        end
        @test data(f)[indexes[check,:]...] == base_array[indexes[check,:]...]
    end
    made_array = convert(Array,data(f))
    if my_rnk == 0
        @test made_array ≈ expected_arr
    end
end
