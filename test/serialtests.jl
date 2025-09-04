using Devito, PyCall, Random, Strided, Test

# configuration!("log-level", "DEBUG")
configuration!("log-level", "WARNING")
configuration!("language", "openmp")
configuration!("mpi", false)

# you need to use when testing locally due to the Libdl startup issue for the nv compiler
configuration!("compiler", get(ENV, "CC", get(ENV, "DEVITO_ARCH", "gcc")))
configuration!("platform", "cpu64")

@testset "configuration" begin
    configuration!("log-level", "INFO")
    @test configuration("log-level") == "INFO"
    configuration!("log-level", "DEBUG")
    c = configuration()
    @test c["log-level"] == "DEBUG"
end

@testset "Grid, n=$n, T=$T" for (n,ex,ori) in ( ( (4,5),(40.0,50.0), (10.0,-10.0) ), ( (4,5,6),(40.0,50.0,60.0),(10.0,0.0,-10.0) ) ), T in (Float32, Float64)
    grid = Grid(shape = n, extent=ex, origin=ori, dtype = T)
    @test size(grid) == n
    @test ndims(grid) == length(n)
    @test eltype(grid) == T
    @show ex
    @show extent(grid)
    @test extent(grid) == ex
    @test origin(grid) == ori
    @test spacing(grid) == ex ./ (n .- 1)
    for i in 1:length(n)
        @test size(grid,i) == n[i]
    end
    halo = []
    for i in 1:length(n)
        push!(halo, [2*i 2*i+1;])
    end
    @test size_with_halo(grid,halo) == size(grid) .+ (sum.(halo)...,)
end

@testset "DevitoArray creation from PyObject n=$n, T=$T" for n in ((5,6),(5,6,7)), T in (Float32, Float64)
    N = length(n)
    array = PyObject(ones(T,n...))
    devito_array = DevitoArray(array)
    @test typeof(devito_array) <: DevitoArray{T,N}
    @test devito_array ≈ ones(T, reverse(n)...)
end

@testset "Function, data_with_halo n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data_with_halo(b)
    @test ndims(b_data) == length(n)

    rand!(b_data)

    b_data_test = data_with_halo(b)
    @test b_data ≈ b_data_test
end

@testset "Function, grid, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    @test grid == Devito.grid(b)
    @test ndims(grid) == length(n)
end

@testset "Function, halo, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    so = 2
    b = Devito.Function(name="b", grid=grid, space_order=so)
    @test ntuple(_->(so,so), length(n)) == halo(b)
end

@testset "Function, ndims, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    @test length(n) == ndims(b)
end

@testset "Function, data, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    b_data = data(b)
    @test ndims(b_data) == length(n)

    copy!(b_data, rand(eltype(grid), size(grid)))

    b_data_test = data(b)
    @test b_data ≈ b_data_test
end

@testset "Function and TimeFunction, space_order, n=$n" for n in ( (4,5), (4,5,6) )
    g = Grid(shape=n)
    for so in (1,2,5,8)
        f = Devito.Function(name="f", grid=g, space_order=so)
        u = Devito.TimeFunction(name="u", grid=g, space_order=so)
        @test space_order(f) == so
        @test space_order(u) == so
    end
end

@testset "TimeFunction, round trip, n=$n" for n in ( (4,5), (4,5,6) )
    g = Grid(shape=n)
    u = Devito.TimeFunction(name="u", grid=g, space_order=4)
    U = Devito.TimeFunction(u.o)
    @test u == U
end

@testset "Constant" begin
    a = Constant(name="a")
    @test isconst(a)
    @test typeof(value(a)) == Float32
    @test value(a) == 0.0
    @test value(a) == data(a)
    value!(a,2.0)
    @test typeof(value(a)) == Float32
    @test value(a) == 2.0
    @test value(a) == data(a)
    value!(a, π)
    @test value(a) == convert(Float32,π)
    @test typeof(convert(Constant,PyObject(a))) == Constant{Float32}
    @test convert(Constant,PyObject(a)) === a

    p = Constant(name="p", dtype=Float64, value=π)
    @test typeof(value(p)) == Float64
    @test value(p) == convert(Float64,π)
    @test data(p) == value(p)
    @test typeof(convert(Constant,PyObject(p))) == Constant{Float64}
    @test convert(Constant,PyObject(p)) === p

    @test_throws  ErrorException("PyObject is not a Constant")  convert(Constant,PyObject(Dimension(name="d")))
end

@testset "TimeFunction, data with halo, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data_with_halo(p)
    @test ndims(p_data) == length(n)+1

    copy!(p_data, rand(eltype(grid), size_with_halo(p)))

    p_data_test = data_with_halo(p)
    @test p_data ≈ p_data_test
end

@testset "TimeFunction, data, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    b = Devito.Function(name="b", grid=grid, space_order=2)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    p_data = data(p)
    @test ndims(p_data) == length(n)+1

    copy!(p_data, rand(eltype(grid), size(p)))

    p_data_test = data(p)
    @test p_data ≈ p_data_test
end

@testset "TimeFunction, grid, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    p = TimeFunction(name="p", grid=grid, time_order=2, space_order=2)
    @test grid == Devito.grid(p)
    @test ndims(grid) == length(n)
end

@testset "TimeFunction, halo, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    so = 2
    p = Devito.TimeFunction(name="p", grid=grid, time_order=2, space_order=so)
    if length(n) == 2
        @test ((so,so),(so,so),(0,0)) == halo(p)
    else
        @test ((so,so),(so,so),(so,so),(0,0)) == halo(p)
    end
end

@testset "TimeFunction, ndims, n=$n" for n in ( (4,5), (4,5,6) )
    grid = Grid(shape = n, dtype = Float32)
    so = 2
    p = Devito.TimeFunction(name="p", grid=grid, time_order=2, space_order=so)
    @test length(n)+1 == ndims(p)
end

@testset "SparseFunction Construction, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    sf = SparseFunction(name="sf", grid=g, npoint=npoint)
    @test typeof(sf) <: SparseFunction{T,1}
    @test sf.o === PyObject(sf)
end

@testset "SparseFunction grid method, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    sf = SparseFunction(name="sf", grid=g, npoint=npoint)
    @test grid(sf) == g
end

@testset "SparseFunction size methods, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    sf = SparseFunction(name="sf", grid=g, npoint=npoint)
    @test size(sf) == (npoint,)
    @test Devito.size_with_inhalo(sf) == (npoint,)
    @test size_with_halo(sf) == (npoint,)
end

@testset "Sparse function coordinates, n=$n" for n in ( (10,11), (10,11,12) )
    grid = Grid(shape=n, dtype=Float32)
    sf = SparseFunction(name="sf", npoint=10, grid=grid)
    @test typeof(coordinates(sf)) <: SubFunction{Float32,2}
    sf_coords = coordinates_data(sf)
    @test isa(sf_coords, Devito.DevitoArray)
    @test size(sf_coords) == (length(n),10)
    x = rand(length(n),10)
    sf_coords .= x
    _sf_coords = coordinates_data(sf)
    @test _sf_coords ≈ x
end

@testset "SparseFunction from PyObject, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    sf = SparseFunction(name="sf", grid=g, npoint=npoint)
    @test SparseFunction(PyObject(sf)) === sf
    stf = SparseTimeFunction(name="stf", grid=g, npoint=npoint, nt=5)
    @test_throws ErrorException("PyObject is not a devito.SparseFunction") SparseFunction(PyObject(stf))
end

@testset "Multidimensional SparseFunction, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    recdim = Dimension(name="recdim")
    nfdim = 7
    fdim = DefaultDimension(name="fdim",  default_value=nfdim)
    sf = SparseFunction(name="sf", grid=g, dimensions=(recdim, fdim), npoint=npoint, shape=(npoint, nfdim))
    @test dimensions(sf) == (recdim, fdim)
    @test size(data(sf)) == (npoint, nfdim)
    @test size(coordinates_data(sf)) == (length(n), npoint)
end

@testset "CoordSlowSparseFunction, T=$T, n=$n, npoint=$npoint" for T in (Float32, Float64), n in ((3,4),(3,4,5)), npoint in (1,5,10)
    g = Grid(shape=n, dtype=T)
    recdim = Dimension(name="recdim")
    nfdim = 7
    fdim = DefaultDimension(name="fdim",  default_value=nfdim)
    sf = CoordSlowSparseFunction(name="sf", grid=g, dimensions=(fdim,recdim), npoint=npoint, shape=(nfdim,npoint))
    @test dimensions(sf) == (fdim, recdim)
    @test size(data(sf)) == (nfdim, npoint)
    @test size(coordinates_data(sf)) == (length(n), npoint)
end

@testset "Sparse time function grid, n=$n, T=$T" for n in ((5,6),(5,6,7)), T in (Float32, Float64)
    N = length(n)
    grd = Grid(shape=n, dtype=T)
    stf = SparseTimeFunction(name="stf", npoint=1, nt=5, grid=grd)
    @test typeof(grid(stf)) <: Grid{T,N}
    @test grid(stf) == grd
end

@testset "Sparse time function coordinates, n=$n" for n in ( (10,11), (10,11,12) )
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
    @test typeof(coordinates(stf)) <: SubFunction{Float32,2}
    stf_coords = coordinates_data(stf)
    @test isa(stf_coords, Devito.DevitoArray)
    @test size(stf_coords) == (length(n),10)
    x = rand(length(n),10)
    stf_coords .= x
    _stf_coords = coordinates_data(stf)
    @test _stf_coords ≈ x
end

@testset "Set Index Writing" begin
    grid = Grid(shape=(11,), dtype=Float32)
    f = Devito.Function(name="f", grid=grid)
    d = data(f)
    d .= 1.0
    op = Operator([Eq(f[1],2.0)],name="indexwrite")
    apply(op)
    @test data(f)[2] == 2.0
end

@testset "Subdomain" begin
    n1,n2 = 5,7
    subdom_mid = SubDomain("subdom_mid", [("middle",1,1), ("middle",2,2)] )
    subdom_lft = SubDomain("subdom_top", [("middle",0,0), ("left",div(n2,2)+1)] )
    subdom_rgt = SubDomain("subdom_bot", [("middle",0,0), ("right",div(n2,2)+1)] )
    subdom_top = SubDomain("subdom_lft", [("left",div(n1,2)+1), ("middle",0,0)] )
    subdom_bot = SubDomain("subdom_rgt", [("right",div(n1,2)+1), ("middle",0,0)] )
    
    grid = Grid(shape=(n1,n2), dtype=Float32, subdomains=(subdom_mid, subdom_lft, subdom_rgt, subdom_top, subdom_bot))
    f0 = Devito.Function(name="f0", grid=grid)
    f1 = Devito.Function(name="f1", grid=grid)
    f2 = Devito.Function(name="f2", grid=grid)
    f3 = Devito.Function(name="f3", grid=grid)
    f4 = Devito.Function(name="f4", grid=grid)
    f5 = Devito.Function(name="f5", grid=grid)
    f6 = Devito.Function(name="f6", grid=grid)
    data(f0) .= 1

    eqns = []
    push!(eqns, Eq(f1,f0,subdomain=subdom_mid))
    push!(eqns, Eq(f2,f0,subdomain=subdom_lft))
    push!(eqns, Eq(f3,f0,subdomain=subdom_rgt))
    push!(eqns, Eq(f4,f0,subdomain=subdom_top))
    push!(eqns, Eq(f5,f0,subdomain=subdom_bot))

    op = Operator(name="op", eqns)
    apply(op)

    _mid = zeros(n1,n2)
    _lft = zeros(n1,n2)
    _rgt = zeros(n1,n2)
    _top = zeros(n1,n2)
    _bot = zeros(n1,n2)

    _mid[2:4,3:5] .= 1
    _lft[:,1:4] .= 1
    _rgt[:,4:7] .= 1
    _top[1:3,:] .= 1
    _bot[3:5,:] .= 1

    @test data(f1) ≈ _mid 
    @test data(f2) ≈ _lft 
    @test data(f3) ≈ _rgt 
    @test data(f4) ≈ _top 
    @test data(f5) ≈ _bot 
end

@testset "Subdomain interior" begin
    n1,n2 = 5,7
    grid = Grid(shape=(n1,n2), dtype=Float32)
    @test Devito.interior(grid) == subdomains(grid)["interior"]
end

@testset "Subdomain equals" begin
    n1,n2 = 5,7
    # note we need to use the same `name` for this work, which is bad form
    sd1 = SubDomain("sd", [("middle",1,1), ("middle",2,2)] )
    sd2 = SubDomain("sd", [("middle",1,1), ("middle",2,2)] )
    @test Base.:(==)(sd1,sd2)
    @test sd1 == sd2
end

@testset "SubDomain Tuple of Vararg{Tuple}" begin
    sd1 = SubDomain("sd", [("middle",1,1), ("middle",2,2)] )
    sd2 = SubDomain("sd", (("middle",1,1), ("middle",2,2)) )
    @test Base.:(==)(sd1,sd2)
    @test sd1 == sd2
end

@testset "Equation Equality, shape=$shape, T=$T" for shape in ((11,11),(11,11,11)), T in (Float32, Float64)
    g = Grid(shape=shape, dtype=T)
    f1 = Devito.Function(name="f1", grid=g, dtype=T)
    f2 = Devito.Function(name="f2", grid=g, dtype=T)
    u1 = TimeFunction(name="u1", grid=g, dtype=T)
    u2 = TimeFunction(name="u2", grid=g, dtype=T)

    eq1 = Eq(f1,1)
    eq2 = Eq(f2,1)
    eq3 = Eq(f1,1)
    eq4 = Eq(u1,1)
    eq5 = Eq(u2,1)
    eq6 = Eq(u1,1)
    eq7 = Eq(u2,u1+f1)
    eq8 = Eq(u2,u1+f1)

    @test eq1 == eq3
    @test eq2 != eq1
    @test eq4 == eq6
    @test eq4 != eq5
    @test eq1 != eq4
    @test eq7 == eq8
end

@testset "Symbolic Min, Max, Size, and Spacing" begin
    x = SpaceDimension(name="x")
    y = SpaceDimension(name="y")
    grid = Grid(shape=(6,11), dtype=Float64, dimensions=(x,y))
    f = Devito.Function(name="f", grid=grid)
    g = Devito.Function(name="g", grid=grid)
    h = Devito.Function(name="h", grid=grid)
    k = Devito.Function(name="k", grid=grid)
    op = Operator([Eq(f,symbolic_max(x)),Eq(g,symbolic_min(y)),Eq(h,symbolic_size(x)),Eq(k,spacing(x))],name="SymMinMax")
    apply(op)
    @test data(f)[1,1] == 5.
    @test data(g)[1,1] == 0.
    @test data(h)[1,1] == 6.
    @test data(k)[1,1] ≈ 1.0/5.0
end

@testset "Min & Max" begin
    grid = Grid(shape=(11,11), dtype=Float64)
    mx = Devito.Function(name="mx", grid=grid)
    mn = Devito.Function(name="mn", grid=grid)
    f = Devito.Function(name="f", grid=grid)
    df = data(f)
    df .= -1.0
    op = Operator([Eq(mn,Min(f,4)),Eq(mx,Max(f,4))],name="minmax")
    apply(op)
    @test data(mn)[5,5] == -1.0
    @test data(mx)[5,5] ==  4
end

@testset "Devito Mathematical Oparations" begin
    # positive only block with equivalent functions in base
    for F in (:sqrt,)
        @eval begin
            vals = (1., 2, 10, 100)
            gr = Grid(shape=(length(vals),), dtype=Float64)
            f = Devito.Function(name="f", grid=gr)
            g = Devito.Function(name="g", grid=gr)
            op = Operator([Eq(g,Devito.$F(f))],name="MathTest")
            data(f) .= vals
            apply(op)
            for i in 1:length(vals)
                @test abs(data(g)[i] - Base.$F(vals[i])) < eps(Float32)
            end
        end
    end
    # positive functions needing base pair specified
    for (F,B) in ((:ln,:log),(:ceiling,:ceil))
        @eval begin
            vals = (1., 2, 10, 100)
            gr = Grid(shape=(length(vals),), dtype=Float64)
            f = Devito.Function(name="f", grid=gr)
            g = Devito.Function(name="g", grid=gr)
            op = Operator([Eq(g,Devito.$F(f))],name="MathTest")
            data(f) .= vals
            apply(op)
            for i in 1:length(vals)
                @test abs(data(g)[i] - Base.$B(vals[i])) < eps(Float32)
            end
        end
    end
    # positive and negative
    for F in (:cos, :sin, :tan, :sinh, :cosh, :tanh, :exp, :floor)
        @eval begin
            vals = (-10, -1, 0, 1., 2, pi, 10)
            gr = Grid(shape=(length(vals),), dtype=Float64)
            f = Devito.Function(name="f", grid=gr)
            g = Devito.Function(name="g", grid=gr)
            op = Operator([Eq(g,Devito.$F(f))],name="MathTest")
            data(f) .= vals
            apply(op)
            for i in 1:length(vals)
                @test abs(data(g)[i] - Base.$F(vals[i])) < eps(Float32)
            end
        end
    end
    # functions needing their own equivalent in base to be specified
    for (F,B) in ((:Abs,:abs),)
        @eval begin
            vals = (-10, -1, 0, 1., 2, pi, 10)
            gr = Grid(shape=(length(vals),), dtype=Float64)
            f = Devito.Function(name="f", grid=gr)
            g = Devito.Function(name="g", grid=gr)
            op = Operator([Eq(g,Devito.$F(f))],name="MathTest")
            data(f) .= vals
            apply(op)
            for i in 1:length(vals)
                @test abs(data(g)[i] - Base.$B(vals[i])) < eps(Float32)
            end
        end
    end  
end

@testset "Unitary Minus" begin
    grid = Grid(shape=(11,), dtype=Float32)
    f = Devito.Function(name="f", grid=grid)
    g = Devito.Function(name="g", grid=grid)
    h = Devito.Function(name="h", grid=grid)
    x = dimensions(f)[1]
    data(f) .= 1.0
    op = Operator([Eq(g,-f),Eq(h,-x)],name="unitaryminus")
    apply(op)
    @show data(f)
    @show data(g)
    for i in 1:length(data(f))
        @test data(g)[i] ≈ -1.0
        @test data(h)[i] ≈ 1-i
    end
end

@testset "Unitary Plus" begin
    grid = Grid(shape=(11,), dtype=Float32)
    f = Devito.Function(name="f", grid=grid)
    g = Devito.Function(name="g", grid=grid)
    h = Devito.Function(name="h", grid=grid)
    x = dimensions(f)[1]
    data(f) .= 1.0
    op = Operator([Eq(g,+f),Eq(h,+x)],name="unitaryplus")
    apply(op)
    for i in 1:length(data(f))
        @test data(g)[i] ≈ 1.0
        @test data(h)[i] ≈ i-1
    end
end

@testset "Mod on Dimensions" begin
    x = SpaceDimension(name="x")
    grid = Grid(shape=(5,), dtype=Float64, dimensions=(x,))
    g = Devito.Function(name="g1", grid=grid)
    eq = Eq(g[x],Mod(x,2))
    op = Operator([eq],name="Mod")
    apply(op)
    for i in 1:5
        @test data(g)[i] == (i-1)%2
    end
end

@testset "PyObject(Dimension)" begin
    x = SpaceDimension(name="x")
    @test PyObject(x) === x.o
end

@testset "Multiply and Divide" begin
    x = SpaceDimension(name="x")
    grid = Grid(shape=(5,), dtype=Float64, dimensions=(x,))
    g1 = Devito.Function(name="g1", grid=grid)
    g2 = Devito.Function(name="g2", grid=grid)
    f1 = Devito.Function(name="f1", grid=grid)
    f2 = Devito.Function(name="f2", grid=grid)
    f3 = Devito.Function(name="f3", grid=grid)
    f4 = Devito.Function(name="f4", grid=grid)
    f5 = Devito.Function(name="f5", grid=grid)
    ffuncs = (f1,f2,f3,f4,f5)
    scalar = 5
    data(g2) .= 5
    data(g1) .= 2
    muleqns = [Eq(f1,g1*g2),Eq(f2,scalar*g1),Eq(f3,g1*scalar),Eq(f4,g1*symbolic_size(x)),Eq(f5,symbolic_size(x)*g1)]
    op = Operator(muleqns,name="Multiplier")
    apply(op)
    for func in ffuncs
        @test data(func)[2] == 10.
    end
    diveqns = [Eq(f1,g1/g2),Eq(f2,scalar/g1),Eq(f3,g1/scalar),Eq(f4,g1/symbolic_size(x)),Eq(f5,symbolic_size(x)/g1)]
    op = Operator(diveqns,name="Divider")
    apply(op)
    for (func,answer) in zip(ffuncs,(2/5,5/2,2/5,2/5,5/2))
        @test data(func)[2] ≈ answer
    end
end

@testset "Symbolic Math" begin
    x = SpaceDimension(name="x")
    y = SpaceDimension(name="y")
    grd = Grid(shape=(5,5), dimensions=(y,x))
    a = Constant(name="a")
    b = Constant(name="b")
    f = Devito.Function(name="f", grid=grd)
    g = Devito.Function(name="g", grid=grd)
    @test a != b
    @test x != y
    @test f != g
    @test x+x+y == 2*x+y
    @test x*y == y*x
    @test x*x == x^2
    @test x+x+a == 2*x+a
    @test a+a+b == 2*a+b
    @test 2*a+b-a == a+b
    @test a*b == b*a
    @test a*a == a^2
    @test f+f+x == 2*f+x
    @test 2*f+x-f == f+x
    @test f*f == f^2
    @test f*g == g*f
    @test f+f+a == 2*f+a
    @test 2*f+a-f == f+a
    @test f+f+g == 2*f+g
    @test 2*f+g-f == f+g
    @test 0*(1+f+a+x) == 0
    @test (1+f+a+x)*0 == 0
end

@testset "Spacing Map" for T in (Float32,Float64)
    grid = Grid(shape=(5,6), dtype=T)
    smap = spacing_map(grid)
    @test typeof(smap) == Dict{PyCall.PyObject, T}
    y,x = dimensions(grid)
    @test smap[spacing(y)] ≈ 1 / (size(grid)[1] - 1)
    @test smap[spacing(x)] ≈ 1 / (size(grid)[2] - 1)
end

@testset "Constants in Operators, T=$T" for T in (Float32,Float64)
    a = Constant(name="a", dtype=T, value=1)
    b = Constant(name="b", dtype=T, value=2)
    grid = Grid(shape=(5,), dtype=T)
    f = Devito.Function(name="f", grid=grid, dtype=T)
    g = Devito.Function(name="g", grid=grid, dtype=T)
    op1 = Operator([Eq(f,a),Eq(g,f+b)],name="op1")
    apply(op1)
    for element in data(f)
        @test element == 1
    end
    for element in data(g)
        @test element == 3
    end
    value!(a,0)
    value!(b,1)
    apply(op1)
    for element in data(f)
        @test element == 0
    end
    for element in data(g)
        @test element == 1
    end
end

@testset "isequal on Devito Objects" begin
    a = Constant(name="a", dtype=Float32)
    b = Constant(name="b", dtype=Float64)
    @test ~isequal(a,b)
    a1 = Constant(a.o)
    @test isequal(a,a1)
    x = SpaceDimension(name="x")
    y = SpaceDimension(name="y")
    g = Grid(shape=(5,4), dtype=Float32, dimensions=(y,x))
    @test spacing(x) ∈ keys(spacing_map(g))
    @test spacing(y) ∈ keys(spacing_map(g))
    y1, x1 = dimensions(g)
    @test isequal(x,x1)
    @test isequal(y,y1)
    f = Devito.Function(name="f", grid=g)
    eq = Eq(f,1)
    op = Operator(eq)
    dict = Dict(f=>true, g=>true, op=>true, eq=>true)
    for entry in (f, eq, op, g)
        @test dict[entry] = true
    end
end

@testset "Math on Dimensions" begin
    x = SpaceDimension(name="x")
    grid = Grid(shape=(5,), dtype=Float64, dimensions=(x,))
    g1 = Devito.Function(name="g1", grid=grid)
    f1 = Devito.Function(name="f1", grid=grid)
    f2 = Devito.Function(name="f2", grid=grid)
    f3 = Devito.Function(name="f3", grid=grid)
    f4 = Devito.Function(name="f4", grid=grid)
    f5 = Devito.Function(name="f5", grid=grid)
    f6 = Devito.Function(name="f6", grid=grid)
    data(g1) .= 1.0

    eq1 = Eq(f1,x+1)
    eq2 = Eq(f2,1+x)
    eq3 = Eq(f3,x+g1)
    eq4 = Eq(f4,g1+x)
    eq5 = Eq(f5,x+1.0*g1)
    eq6 = Eq(f6,1.0*g1+x)
    opl = Operator([eq1,eq3,eq5],name="Left")
    opr = Operator([eq2,eq4,eq6],name="Right")
    apply(opl)
    apply(opr)

    for f in (f1,f2,f3,f4,f5,f6)
        df = data(f)
        for i in 1:length(df)
            @test df[i] == i
        end
    end
end

@testset "Devito Dimension Constructors" begin
    attribtes = (:is_Dimension, :is_Space, :is_Time, :is_Default, :is_Custom, :is_Derived, :is_NonlinearDerived, :is_Sub, :is_Conditional, :is_Stepping, :is_Modulo, :is_Incr)
    a = Dimension(name="a")
    b = SpaceDimension(name="b")
    c = TimeDimension(name="c")
    d = SteppingDimension(name="d",parent=c)
    @test parent(d) == c
    e = DefaultDimension(name="e")
    f = ConditionalDimension(name="f", parent=c, factor=2)
    @test parent(f) == c
    for (dim,attribute) in ((_dim,_attribute) for _dim in (a,b,c,d,e,f) for _attribute in attribtes)
        @eval begin
            @test $attribute($dim) == $dim.o.$attribute
        end
    end
    for _dim in (a,b,c,d,e,f)
        @test typeof(dimension(PyObject(_dim))) == typeof(_dim)
        @test dimension(PyObject(_dim)) === _dim
    end
    # tests for ErrorExceptiosn
    grd = Grid(shape=(5,4))
    @test_throws ErrorException("not implemented")  dimension(PyObject(grd))
end

@testset "Devito SubDimensions" begin
    d = SpaceDimension(name="d")
    dl = SubDimensionLeft(name="dl", parent=d, thickness=2)
    dr = SubDimensionRight(name="dr", parent=d, thickness=3)
    dm = SubDimensionMiddle(name="dm", parent=d, thickness_left=2, thickness_right=3)
    for subdim in (dl,dr,dm)
        @test parent(subdim) == d
        @test PyObject(subdim) == subdim.o
    end
    @test (thickness(dl)[1].value, thickness(dl)[2].value) == (2, nothing)
    @test (thickness(dr)[1].value, thickness(dr)[2].value) == (nothing, 3)
    @test (thickness(dm)[1].value, thickness(dr)[2].value) == (2, 3)
end

@testset "Devito stepping dimension" begin
    grid = Grid(shape=(5,5),origin=(0.,0.),extent=(1.,1.))
    f = TimeFunction(grid=grid,space_order=8,time_order=2,name="f")
    @test stepping_dim(grid) == time_dim(f)
    @test stepping_dim(grid) != time_dim(grid)
    @test stepping_dim(grid).o.is_Stepping
end

@testset "Sparse Function data with halo npoint=$npoint" for npoint in (1,5)
    grid = Grid(shape=(5,5))
    sf = SparseFunction(name="sf", grid=grid, npoint=npoint)
    for i in 1:npoint
        data(sf)[i] = Float32(i)
    end
    for i in 1:npoint
        @test data_with_halo(sf)[i] == Float32(i)
    end
end

@testset "Sparse Time Function data with halo npoint=$npoint" for npoint in (1,5)
    grid = Grid(shape=(5,5))
    nt = 10
    stf = SparseTimeFunction(name="stf", grid=grid, npoint=npoint, nt=nt)
    for i in 1:npoint
        data(stf)[i,:] .= Float32(i) * [1:nt;]
    end
    for i in 1:npoint
        @test data_with_halo(stf)[i,:] ≈ Float32(i) * [1:nt;]
    end
end

@testset "Sparse Time Function Inject and Interpolate" begin
    dt = 0.01
    nt = 101
    time_range = 0.0f0:dt:dt*(nt-1)

    grid = Grid(shape=(5,5),origin=(0.,0.),extent=(1.,1.))
    p = TimeFunction(grid=grid,space_order=8,time_order=2,name="p")
    y,x,t = dimensions(p)
    dt = step(time_range)
    smap = spacing_map(grid)
    smap[spacing(t)] = dt

    src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=nt)
    @test typeof(dimensions(src)[1]) == Dimension
    coords =  [0; 0.5]
    src_coords = coordinates_data(src)
    src_coords .= coords
    src_data = data(src)
    src_data .= reshape(1e3*Base.sin.(time_range .* (3*pi/2)),1,:)
    pupdate = Eq(forward(p),1+p)
    src_term = inject(src; field=forward(p), expr=src*spacing(t)^2)

    rec = SparseTimeFunction(name="rec", grid=grid, npoint=2, nt=nt)
    rec_coords = coordinates_data(rec)
    rec_coords[:,1] .= coords
    rec_coords[:,2] .= reverse(coords)
    rec_term = interpolate(rec, expr=p)

    op = Operator([pupdate,src_term,rec_term],name="SparseInjectInterp", subs=smap)
    apply(op,time_M=nt-1)
    @test data(p)[3,1,end-1] ≈ (nt-1) + sum(src_data[1:end-1])*dt^2
    @test data(rec)[1,end] ≈ (nt-1) + sum(src_data[1:end-1])*dt^2
    @test data(rec)[2,end] ≈ (nt-1)
end

@testset "Sparse Function Inject and Interpolate" begin
    grid = Grid(shape=(5,5),origin=(0.,0.),extent=(1.,1.))
    f = Devito.Function(grid=grid,space_order=8,time_order=2,name="f")
    y,x = dimensions(f)

    src = SparseFunction(name="src", grid=grid, npoint=1)
    @test typeof(dimensions(src)[1]) == Dimension
    coords =  [0; 0.5]
    src_coords = coordinates_data(src)
    src_coords .= coords
    src_data = data(src)
    src_data .= 1
    src_term = inject(src; field=f, expr=src)

    rec = SparseFunction(name="rec", grid=grid, npoint=2)
    rec_coords = coordinates_data(rec)
    rec_coords[:,1] .= coords
    rec_coords[:,2] .= reverse(coords)
    rec_term = interpolate(rec, expr=f)

    op = Operator([src_term,rec_term],name="SparseInjectInterp")
    apply(op)
    @test data(f)[3,1] == 1.0
    # check that this was the only place where f became nonzero
    data(f)[3,1] = 0.0
    @test data(f) ≈ zeros(Float32,size(f)...)
    @test data(rec)[1] == 1
    @test data(rec)[2] == 0
end

# dxl/dxr implement Fornberg 1988 table 3, derivative order 1, order of accuracy 2
@testset "Left and Right Derivatives" begin
    fornberg = Float64[-3/2, 2.0, -1/2]
    n = 5
    grid = Grid(shape=(n),extent=(n-1,))
    x = dimensions(grid)[1]
    fff = Devito.Function(name="fff", grid=grid, space_order=2)
    fxl = Devito.Function(name="fxl", grid=grid, space_order=2)
    fxr = Devito.Function(name="fxr", grid=grid, space_order=2)
    data(fff)[div(n,2)+1] = 1.
    eq1 = Eq(fxl, dxl(fff))
    eq2 = Eq(fxr, dxr(fff))
    op = Operator([eq1,eq2],name="Derivatives")
    apply(op)
    @test data(fxl)[3:5] ≈ -1 .* fornberg
    @test data(fxr)[1:3] ≈ +1 .* reverse(fornberg)
end

@testset "Derivative Operator and Mixed Derivatives" begin
    grid = Grid(shape=(12,16))
    f  = Devito.Function(grid=grid, name="f", space_order=8)
    y, x = dimensions(f)
    g1 = Devito.Function(grid=grid, name="g1", space_order=8)
    g2 = Devito.Function(grid=grid, name="g2", space_order=8)
    h1 = Devito.Function(grid=grid, name="h1", space_order=8)
    h2 = Devito.Function(grid=grid, name="h2", space_order=8)
    j1 = Devito.Function(grid=grid, name="j1", space_order=8)
    j2 = Devito.Function(grid=grid, name="j2", space_order=8)
    k1 = Devito.Function(grid=grid, name="k1", space_order=8)
    k2 = Devito.Function(grid=grid, name="k2", space_order=8)

    data(f) .= rand(Float32, size(f)...)
    eq1a = Eq(g1, dx2(f))
    eq1b = Eq(g2, Derivative(f, (x,2)))
    eq2a = Eq(h1, dy2(f))
    eq2b = Eq(h2, Derivative(f, (y,2)))
    eq3a = Eq(j1, dxdy(f))
    eq3b = Eq(j2, Derivative(f, y, x))
    eq4a = Eq(k1, dy(dx2(f)))
    eq4b = Eq(k2, Derivative(f, x, y, deriv_order=(2,1)))
    
    derivop = Operator([eq1a, eq1b, eq2a, eq2b, eq3a, eq3b, eq4a, eq4b], name="derivOp")
    
    apply(derivop)

    @test sum(abs.(data(g1))) > 0
    @test sum(abs.(data(h1))) > 0
    @test sum(abs.(data(j1))) > 0
    @test sum(abs.(data(k1))) > 0

    @test data(g1) ≈ data(g2)
    @test data(h1) ≈ data(h2)
    @test data(j1) ≈ data(j2)
    @test data(k1) ≈ data(k2)
end

@testset "Derivatives on Constants" begin
    for x in (Constant(name="a", value=2), Constant(name="b", dtype=Float64, value=2), 1, -1.0, π)
        @test dx(x) == 0
        @test dxl(x) == 0
        @test dxr(x) == 0
        @test dy(x) == 0
        @test dyl(x) == 0
        @test dyr(x) == 0
        @test dz(x) == 0
        @test dzl(x) == 0
        @test dzr(x) == 0
        @test dx2(x) == 0
        @test dy2(x) == 0
        @test Derivative(x) == 0
        @test dx(dx(x)+1) == 0
        @test dxl(dxl(x)+1) == 0
        @test dxr(dxr(x)+1) == 0
        @test dy(dy(x)+1) == 0
        @test dyl(dyl(x)+1) == 0
        @test dyr(dyr(x)+1) == 0
        @test dz(dz(x)+1) == 0
        @test dzl(dzl(x)+1) == 0
        @test dzr(dzr(x)+1) == 0
        @test dx2(dx2(x)+1) == 0
        @test dy2(dy2(x)+1) == 0
    end
end

@testset "Derivatives on dimensions not in a function, T=$T" for T in (Float32,Float64)
    x = SpaceDimension(name="x")
    grid = Grid(shape=(5,), dimensions=(x,), dtype=T)
    f = Devito.Function(name="f", grid=grid, dtype=T)
    u = Devito.TimeFunction(name="u", grid=grid, dtype=T)
    a = Constant(name="a", dtype=T, value=2)
    b = Constant(name="b", dtype=T, value=2)
    for func in (f,u)
        @test dy(func) == 0
        @test dyl(func) == 0
        @test dyr(func) == 0
        @test dz(func) == 0
        @test dzl(func) == 0
        @test dzr(func) == 0
        @test dy(b*func+a-1) == 0
        @test dyl(b*func+a-1) == 0
        @test dyr(b*func+a-1) == 0
        @test dz(b*func+a-1) == 0
        @test dzl(b*func+a-1) == 0
        @test dzr(b*func+a-1) == 0
    end
end

@testset "Conditional Dimension Subsampling" begin
    size, factr = 17, 4
    i = Devito.SpaceDimension(name="i")
    grd = Grid(shape=(size,),dimensions=(i,))
    ci = ConditionalDimension(name="ci", parent=i, factor=factr)
    @test parent(ci) == i
    g = Devito.Function(name="g", grid=grd, shape=(size,), dimensions=(i,))
    f = Devito.Function(name="f", grid=grd, shape=(div(size,factr),), dimensions=(ci,))
    op = Operator([Eq(g, i), Eq(f, g)],name="Conditional")
    apply(op)
    for j in 1:div(size,factr)
        @test data(f)[j] == data(g)[(j-1)*factr+1]
    end
end

@testset "Conditional Dimension Honor Condition" begin
    # configuration!("log-level", "DEBUG")
    # configuration!("opt", "noop")
    # configuration!("jit-backdoor", false)
    # configuration!("jit-backdoor", true)
    # @show configuration()
    x = Devito.SpaceDimension(name="x")
    y = Devito.SpaceDimension(name="y")
    grd = Grid(shape=(5,5),dimensions=(y,x),extent=(4,4))
    f1 = Devito.Function(name="f1", grid=grd, space_order=0, is_transient=true)
    f2 = Devito.Function(name="f2", grid=grd, space_order=0, is_transient=true)
    f3 = Devito.Function(name="f3", grid=grd, space_order=0, is_transient=true)
    g = Devito.Function(name="g", grid=grd, space_order=0, is_transient=true)
    data(f1) .= 1.0
    data(f2) .= 1.0
    data(g)[:,3:end] .= 1.0
    ci1 = ConditionalDimension(name="ci1", parent=y, condition=And(Ne(g, 0), Lt(y, 2)))
    ci2 = ConditionalDimension(name="ci2", parent=y, condition=And(Ne(g, 0), Le(y, 2)))
    # Note, the order of dimensions feeding into parent seems to matter.  
    # If the grid dimensions were (x,y) 
    # the above conditional dimension would cause a compilation error on the operator at runtime.
    # write(stdout,"\n")
    # @show data(f1)
    # @show data(f2)
    # @show data(f3)
    # @show data(g)

    eq1 = Eq(f1, f1+g, implicit_dims=ci1)
    eq2 = Eq(f2, f2+g, implicit_dims=ci2)
    eq3 = Eq(f3, f2-f1)
    op = Operator([eq1,eq2,eq3], name="Implicit")
    # apply(op, nthreads=2)
    apply(op)

    # write(stdout,"\n")
    # @show data(f1)
    # @show data(f2)
    # @show data(f3)
    # @show data(g)

    # write(stdout,"\n")
    # @show view(DevitoArray{Float32,2}(f1.o."_data_allocated"), localindices(f1)...)

    @test data(f1)[1,1] == 1.0
    @test data(f1)[1,3] == 2.0
    @test data(f1)[1,5] == 2.0
    @test data(f1)[2,2] == 1.0
    @test data(f1)[2,3] == 2.0
    @test data(f1)[3,3] == 1.0
    @test data(f1)[5,1] == 1.0
    @test data(f1)[5,3] == 1.0
    @test data(f1)[5,5] == 1.0

    @test data(f3)[1,1] == 0.0
    @test data(f3)[3,1] == 0.0
    @test data(f3)[3,2] == 0.0
    @test data(f3)[3,3] == 1.0
    @test data(f3)[3,4] == 1.0
    @test data(f3)[3,5] == 1.0
    @test data(f3)[2,3] == 0.0
    @test data(f3)[2,4] == 0.0
    @test data(f3)[2,5] == 0.0
    @test data(f3)[4,3] == 0.0
    @test data(f3)[4,4] == 0.0
    @test data(f3)[4,5] == 0.0
end

@testset "Conditional Dimension factor" begin
    size, factr = 17, 4
    i = Devito.SpaceDimension(name="i")
    grd = Grid(shape=(size,),dimensions=(i,))
    ci = ConditionalDimension(name="ci", parent=i, factor=factr)
    @test factor(ci) == ci.o.factor
end

@testset "Conditional Dimension equality" begin
    size, factr = 17, 4
    i = Devito.SpaceDimension(name="i")
    grd = Grid(shape=(size,),dimensions=(i,))
    ci1 = ConditionalDimension(name="ci", parent=i, factor=factr)
    ci2 = ConditionalDimension(name="ci", parent=i, factor=factr)
    @test ci1 == ci2
end

@testset "Retrieve time_dim" begin
    g = Grid(shape=(5,5))
    @test time_dim(g) == dimension(g.o.time_dim)
    t = TimeDimension(name="t")
    f = TimeFunction(name="f",time_dim=t,grid=g)
    @test time_dim(f) == t
    @test time_dim(f) == dimensions(f)[end]
end

@testset "Dimension ordering in Function and Time Function Constuction, n=$n" for n in ((5,6),(4,5,6))
    g = Grid(shape=n)
    dims = dimensions(g)
    f = Devito.Function(name="f", grid=g, dimensions=dims)
    @test dimensions(f) == dims
    t = stepping_dim(g)
    u = TimeFunction(name="u", grid=g, dimensions=(dims...,t))
    @test typeof(dimensions(u)[end]) == Devito.SteppingDimension
    @test dimensions(u) == (dims...,t)
end

@testset "Dimension ordering in SparseTimeFunction construction, n=$n" for n in ((5,6),(4,5,6))
    g = Grid(shape=n)
    p = Dimension(name="p")
    t = time_dim(g)
    dims = (p,t)
    stf = SparseTimeFunction(name="stf", dimensions=dims, npoint=5, nt=4, grid=g)
    @test dimensions(stf) == dims
end

@testset "Time Derivatives" begin
    grd = Grid(shape=(5,5))
    t = TimeDimension(name="t")
    f1 = TimeFunction(name="f1",grid=grd,time_order=2,time_dim=t)
    f2 = TimeFunction(name="f2",grid=grd,time_order=2,time_dim=t)
    f3 = TimeFunction(name="f3",grid=grd,time_order=2,time_dim=t)
    data(f1)[:,:,1] .= 0
    data(f1)[:,:,2] .= 1
    data(f1)[:,:,3] .= 4
    smap = spacing_map(grd)
    t_spacing = 0.5
    smap[spacing(t)] = t_spacing
    op = Operator([Eq(f2,dt(f1)),Eq(f3,dt2(f1))],name="DerivTest",subs=smap)
    apply(op)
    @test data(f2)[3,3,2] == (data(f1)[3,3,3] - data(f1)[3,3,2])/t_spacing
    @test data(f3)[3,3,2] == (data(f1)[3,3,3] - 2*data(f1)[3,3,2] + data(f1)[3,3,1] )/t_spacing^2
end

@testset "Time subs" begin
    grd = Grid(shape=(5,5))
    y,x = dimensions(grd)
    f = TimeFunction(name="f",grid=grd)
    @test subs(f,Dict(x => x+1)) == subs(f.o,Dict(x => x+1))
end

@testset "nsimplify" begin
    @test nsimplify(0) == 0
    @test nsimplify(-1) == -1
    @test nsimplify(1) == 1
    @test nsimplify(π; tolerance=0.1) == nsimplify(22/7)
    @test nsimplify(π) != nsimplify(π; tolerance=0.1)
    g = Grid(shape=(5,))
    x = dimensions(g)[1]
    @test nsimplify(x+1) == x+1
    @test nsimplify(1+x) == x+1
end

@testset "solve" begin
    g = Grid(shape=(11,11))
    u = TimeFunction(grid=g, name="u", time_order=2, space_order=8)
    v = TimeFunction(grid=g, name="v", time_order=2, space_order=8)
    for k in 1:size(data(u))[3], j in 1:size(data(u))[2], i in 1:size(data(u))[1]
        data(u)[i,j,k] = k*0.02 + i*i*0.01 - j*j*0.03
    end
    data(v) .= data(u)
    y,x,t = dimensions(u)
    pde = dt2(u) - (dx(dx(u)) + dy(dy(u))) - dy(dx(u))
    solved = Eq(forward(u),solve(pde, forward(u)))
    eq =  Eq(forward(v), (2 * v - backward(v)) + spacing(t)^2 * ( dx(dx(v)) - dy(dx(v)) + dy(dy(v))))
    smap = spacing_map(g)
    smap[spacing(t)] = 0.001
    op = Operator([solved,eq],subs=smap,name="solve")
    apply(op,time_M=5)
    @test data(v) ≈ data(u)
end

@testset "name" begin
    a = Constant(name="a")
    @test name(a) == "a"
    x = SpaceDimension(name="x")
    @test name(x) == "x"
    t = TimeDimension(name="t")
    @test name(t) == "t"
    t1 = ConditionalDimension(name="t1", parent=t, factor=2)
    @test name(t1) == "t1"
    time = SteppingDimension(name="time", parent=t)
    @test name(time) == "time"
    grid = Grid(shape=(5,))
    f = Devito.Function(name="f", grid=grid)
    @test name(f) == "f"
    u = Devito.TimeFunction(name="u", grid=grid)
    @test name(u) == "u"
    sf = SparseFunction(name="sf", npoint=1, grid=grid)
    @test name(sf) == "sf"
    stf = SparseTimeFunction(name="stf", npoint=1, nt=10, grid=grid)
    @test name(stf) == "stf"
    op = Operator(Eq(f,1), name="op")
    @test name(op) == "op"
end

# jkw: had to switch to py"repr" to get string representation of PyObject
# something must have changes somewhere as we can no longer directly compare like `g == evaluate(h)``
@testset "subs" begin
    grid = Grid(shape=(5,5,5))
    dims = dimensions(grid)
    for staggered in ((dims[1],),(dims[2],),(dims[3],),dims[1:2],dims[2:3],(dims[1],dims[3]),dims)
        f = Devito.Function(name="f", grid=grid, staggered=staggered)
        for d in staggered
            stagdict1 = Dict()
            stagdict2 = Dict()
            stagdict1[d] = d-spacing(d)/2
            stagdict2[d] = d-spacing(d)
            h = f
            g = f
            h = subs(h,stagdict1)
            g = .5 * (g + subs(g,stagdict2))
            sg = py"repr"(g)
            sh = py"repr"(evaluate(h))
            @show sg, sh, sg == sh
            @test sg == sh
        end
    end 
end

@testset "ccode" begin
    grd = Grid(shape=(5,5))
    f = Devito.Function(grid=grd, name="f")
    op = Operator(Eq(f,1),name="ccode")
    @test ccode(op) === nothing
    ccode(op; filename="/tmp/ccode.cc")
    code = open(f->read(f, String),"/tmp/ccode.cc")
    @test typeof(code) == String
    @test code != ""
end

@testset "Operator default naming" begin
    grid1 = Devito.Grid(shape=(2,2), origin=(0,0), extent=(1,1), dtype=Float32)
    f = Devito.Function(name="f", grid=grid1, space_order=4)

    op = Operator([Eq(f,1)]; name="foo")
    @test name(op) == "foo"
    op = Operator([Eq(f,1)])
    @test name(op) == "Kernel"
    op = Operator(Eq(f,1))
    @test name(op) == "Kernel"
    op = Operator( (Eq(f,1), Eq(f,1)))
    @test name(op) == "Kernel"
    op = Operator( [Eq(f,1), Eq(f,1)])
    @test name(op) == "Kernel"
    op = Operator(Eq(f,1), opt="advanced")
    @test name(op) == "Kernel"
    op = Operator( (Eq(f,1), Eq(f,1)), opt="advanced")
    @test name(op) == "Kernel"
    op = Operator( [Eq(f,1), Eq(f,1)], opt="advanced")
    @test name(op) == "Kernel"
end

@testset "operator PyObject convert" begin
    grid = Grid(shape=(3,4))
    f = Devito.Function(name="f", grid=grid)
    op = Operator(Eq(f,1), name="ConvertOp")
    @test typeof(convert(Operator, PyObject(op))) == Operator
    @test  convert(Operator, PyObject(op)) === op
    @test_throws ErrorException("PyObject is not an operator") convert(Operator, PyObject(f)) 
end

@testset "in_range throws out of range error" begin
    @test_throws ErrorException("Outside Valid Ranges") Devito.in_range(10, ([1:5],[6:9]))
end

@testset "Serial inner halo methods, n=$n, space_order=$space_order" for n in ((3,4),(3,4,5)), space_order in (1,2,4)
    grd = Grid(shape=n)
    N = length(n)
    time_order = 2
    nt = 11
    npoint=6
    f = Devito.Function(name="f", grid=grd, space_order=space_order)
    u = TimeFunction(name="u", grid=grd, space_order=space_order, time_order=2)
    sf = SparseFunction(name="sf", grid=grd, npoint=npoint)
    stf = SparseTimeFunction(name="stf", grid=grd, npoint=npoint, nt=nt)
    for func in (f,u,sf,stf)
        data(func) .= 1.0
    end
    halo_n = (2*space_order) .+ n
    @test size(data_with_inhalo(f)) == halo_n
    @test size(data_with_inhalo(u)) == (halo_n...,time_order+1)
    @test size(data_with_inhalo(sf)) == (npoint,)
    @test size(data_with_inhalo(stf)) == (npoint,nt)
    haloed_f = zeros(Float32, halo_n...)
    haloed_u = zeros(Float32, halo_n...,time_order+1)
    if N == 2
        haloed_f[1+space_order:end-space_order,1+space_order:end-space_order] .= 1.0
        haloed_u[1+space_order:end-space_order,1+space_order:end-space_order,:] .= 1.0
    else
        haloed_f[1+space_order:end-space_order,1+space_order:end-space_order,1+space_order:end-space_order] .= 1.0
        haloed_u[1+space_order:end-space_order,1+space_order:end-space_order,1+space_order:end-space_order,:] .= 1.0
    end
    @test data_with_inhalo(f) ≈ haloed_f
    @test data_with_inhalo(u) ≈ haloed_u
    @test data_with_inhalo(sf) ≈ ones(Float32, npoint)
    @test data_with_inhalo(stf) ≈ ones(Float32, npoint, nt)
end

@testset "Buffer construction and use, buffer size = $value" for value in (1,2,4)
    b = Buffer(value)
    @test typeof(b) == Buffer
    shp = (5,6)
    grd = Grid(shape=shp)
    u = TimeFunction(name="u", grid=grd, save=b)
    @test size(u) == (shp...,value)
end

@testset "Generate Function from PyObject, n=$n" for n in ((3,4),(3,4,5))
    g = Grid(shape=n)
    f1 = Devito.Function(name="f1", grid=g)
    f2 = Devito.Function(PyObject(f1))
    @test isequal(f1, f2)
    # try to make Functions from non-function objects
    u = TimeFunction(name="u", grid=g)
    @test_throws ErrorException("PyObject is not a devito.Function") Devito.Function(PyObject(u))
    c = Constant(name="c")
    @test_throws ErrorException("PyObject is not a devito.Function") Devito.Function(PyObject(c))
    s = SparseFunction(name="s", grid=g, npoint=5)
    @test_throws ErrorException("PyObject is not a devito.Function") Devito.Function(PyObject(s))
    st = SparseTimeFunction(name="st", grid=g, npoint=5, nt=10)
    @test_throws ErrorException("PyObject is not a devito.Function") Devito.Function(PyObject(st))
    @test_throws ErrorException("PyObject is not a devito.Function") Devito.Function(PyObject(1))
end

@testset "Generate SparseTimeFunction from PyObject, n=$n" for n in ((3,4),(3,4,5))
    g = Grid(shape=n)
    s1 = SparseTimeFunction(name="s1", grid=g, nt=10, npoint=5)
    s2 = SparseTimeFunction(PyObject(s1))
    @test isequal(s1, s2)
    # try to make Functions from non-function objects
    f = Devito.Function(name="f", grid=g)
    @test_throws ErrorException("PyObject is not a devito.SparseTimeFunction") SparseTimeFunction(PyObject(f))
    u = TimeFunction(name="u", grid=g)
    @test_throws ErrorException("PyObject is not a devito.SparseTimeFunction") SparseTimeFunction(PyObject(u))
    c = Constant(name="c")
    @test_throws ErrorException("PyObject is not a devito.SparseTimeFunction") SparseTimeFunction(PyObject(c))
    s = SparseFunction(name="s", grid=g, npoint=5)
    @test_throws ErrorException("PyObject is not a devito.SparseTimeFunction") SparseTimeFunction(PyObject(s))
    @test_throws ErrorException("PyObject is not a devito.SparseTimeFunction") SparseTimeFunction(PyObject(1))
end

@testset "Indexed Data n=$n, T=$T, space_order=$so" for n in ((3,4), (3,4,5)), T in (Float32, Float64), so in (4,8)
    g = Grid(shape=n, dtype=T)
    f = Devito.Function(name="f", grid=g, space_order=so)
    fi = indexed(f)
    @test typeof(fi) <: Devito.IndexedData
    fi_index = fi[(n .- 1 )...]
    @show fi_index
    x = dimensions(g)[end]
    fd_index = fi[(n[1:end-1] .- 2)..., x]
    @test typeof(fi_index) <: Devito.Indexed
    @test typeof(fd_index) <: Devito.Indexed
    op = Operator([Eq(fi_index, 1), Eq(fd_index, 2)])
    apply(op)
    @test data(f)[(n .- 1 )...] == 1
    data(f)[(n .- 1 )...] = 0
    @test data(f)[(n[1:end-1] .- 2)...,:] ≈ 2 .* ones(T, n[end])
    data(f)[(n[1:end-1] .- 2)...,:] .= 0
    @test data(f) ≈ zeros(T, n...)
    @test PyObject(fi) == fi.o
end

@testset "Function Inc, shape=$n" for n in ((4,5),(6,7,8),)
    grid = Grid(shape=n)
    A = Devito.Function(name="A", grid=grid)
    v = Devito.Function(name="v", grid=grid, shape=size(grid)[1:end-1], dimensions=dimensions(grid)[1:end-1])
    b = Devito.Function(name="b", grid=grid, shape=(size(grid)[end],), dimensions=(dimensions(grid)[end],))
    data(v) .= 1.0
    data(A) .= reshape([1:prod(size(grid));],size(grid)...)
    op = Operator([Inc(b, A*v)], name="inctest")
    apply(op)
    @test data(b)[:] ≈ sum(data(A), dims=Tuple([1:length(n)-1;]))[:]
end

@testset "derivative shorthand dxl,dyl,dzl" begin
    shape=(11,12,13)
    grid = Grid(shape=shape, dtype=Float32)
    f = Devito.Function(name="f", grid=grid, space_order=8)
    fx1 = Devito.Function(name="fx1", grid=grid, space_order=8)
    fx2 = Devito.Function(name="fx2", grid=grid, space_order=8)
    fy1 = Devito.Function(name="fy1", grid=grid, space_order=8)
    fy2 = Devito.Function(name="fy2", grid=grid, space_order=8)
    fz1 = Devito.Function(name="fz1", grid=grid, space_order=8)
    fz2 = Devito.Function(name="fz2", grid=grid, space_order=8)
    z,y,x = dimensions(f)
    data(f) .= rand(Float32,shape)
    eqx1 = Eq(fx1,dxc(f))
    eqx2 = Eq(fx2, Derivative(f, x, size="left", deriv_order=1))
    eqy1 = Eq(fy1,dyc(f))
    eqy2 = Eq(fy2, Derivative(f, y, size="left", deriv_order=1))
    eqz1 = Eq(fz1,dzc(f))
    eqz2 = Eq(fz2, Derivative(f, z, size="left", deriv_order=1))
    op = Operator([eqx1,eqx2,eqy1,eqy2,eqz1,eqz2], name="op")
    apply(op)
    @test maximum(abs,data(fx1)) > 0
    @test maximum(abs,data(fx2)) > 0
    @test maximum(abs,data(fy1)) > 0
    @test maximum(abs,data(fy2)) > 0
    @test maximum(abs,data(fz1)) > 0
    @test maximum(abs,data(fz2)) > 0
    @test isapprox(data(fx1), data(fx2))
    @test isapprox(data(fy1), data(fy2))
    @test isapprox(data(fz1), data(fz2))
end

@testset "derivative shorthand dxr,dyr,dzr" begin
    shape=(11,12,13)
    grid = Grid(shape=shape, dtype=Float32)
    f = Devito.Function(name="f", grid=grid, space_order=8)
    fx1 = Devito.Function(name="fx1", grid=grid, space_order=8)
    fx2 = Devito.Function(name="fx2", grid=grid, space_order=8)
    fy1 = Devito.Function(name="fy1", grid=grid, space_order=8)
    fy2 = Devito.Function(name="fy2", grid=grid, space_order=8)
    fz1 = Devito.Function(name="fz1", grid=grid, space_order=8)
    fz2 = Devito.Function(name="fz2", grid=grid, space_order=8)
    z,y,x = dimensions(f)
    data(f) .= rand(Float32,shape)
    eqx1 = Eq(fx1,dxc(f))
    eqx2 = Eq(fx2, Derivative(f, x, size="right", deriv_order=1))
    eqy1 = Eq(fy1,dyc(f))
    eqy2 = Eq(fy2, Derivative(f, y, size="right", deriv_order=1))
    eqz1 = Eq(fz1,dzc(f))
    eqz2 = Eq(fz2, Derivative(f, z, size="right", deriv_order=1))
    op = Operator([eqx1,eqx2,eqy1,eqy2,eqz1,eqz2], name="op")
    apply(op)
    @test maximum(abs,data(fx1)) > 0
    @test maximum(abs,data(fx2)) > 0
    @test maximum(abs,data(fy1)) > 0
    @test maximum(abs,data(fy2)) > 0
    @test maximum(abs,data(fz1)) > 0
    @test maximum(abs,data(fz2)) > 0
    @test isapprox(data(fx1), data(fx2))
    @test isapprox(data(fy1), data(fy2))
    @test isapprox(data(fz1), data(fz2))
end

@testset "derivative shorthand dxc,dyc,dzc" begin
    shape=(11,12,13)
    grid = Grid(shape=shape, dtype=Float32)
    f = Devito.Function(name="f", grid=grid, space_order=8)
    fx1 = Devito.Function(name="fx1", grid=grid, space_order=8)
    fx2 = Devito.Function(name="fx2", grid=grid, space_order=8)
    fy1 = Devito.Function(name="fy1", grid=grid, space_order=8)
    fy2 = Devito.Function(name="fy2", grid=grid, space_order=8)
    fz1 = Devito.Function(name="fz1", grid=grid, space_order=8)
    fz2 = Devito.Function(name="fz2", grid=grid, space_order=8)
    z,y,x = dimensions(f)
    data(f) .= rand(Float32,shape)
    eqx1 = Eq(fx1,dxc(f))
    eqx2 = Eq(fx2, Derivative(f, x, size="centered", deriv_order=1))
    eqy1 = Eq(fy1,dyc(f))
    eqy2 = Eq(fy2, Derivative(f, y, size="centered", deriv_order=1))
    eqz1 = Eq(fz1,dzc(f))
    eqz2 = Eq(fz2, Derivative(f, z, size="centered", deriv_order=1))
    op = Operator([eqx1,eqx2,eqy1,eqy2,eqz1,eqz2], name="op")
    apply(op)
    @test maximum(abs,data(fx1)) > 0
    @test maximum(abs,data(fx2)) > 0
    @test maximum(abs,data(fy1)) > 0
    @test maximum(abs,data(fy2)) > 0
    @test maximum(abs,data(fz1)) > 0
    @test maximum(abs,data(fz2)) > 0
    @test isapprox(data(fx1), data(fx2))
    @test isapprox(data(fy1), data(fy2))
    @test isapprox(data(fz1), data(fz2))
end

@testset "laplacian" begin
    shape=(11,21,31)
    grid = Grid(shape=shape, dtype=Float32, origin=(0,0,0), extent=shape .- 1)
    f = Devito.Function(name="f", grid=grid, space_order=8)
    g = Devito.Function(name="g", grid=grid, space_order=8)
    h = Devito.Function(name="h", grid=grid, space_order=8)
    z,y,x = dimensions(f)
    a,b,c = 1.2,1.7,2.3
    df = data_with_halo(f)
    n1,n2,n3 = size(df)
    for k1 ∈ 1:n1
        for k2 ∈ 1:n2
            for k3 ∈ 1:n3
                df[k1,k2,k3] = 1/2 * (a * k1^2 + b * k2^2 + c * k3^2)
            end
        end
    end
    eq1 = Eq(g,laplacian(f))
    eq2 = Eq(h,dx(dx(f,x0=x-spacing(x)),x0=x+spacing(x)) + dy(dy(f,x0=y-spacing(y)),x0=y+spacing(y)) + dz(dz(f,x0=z-spacing(z)),x0=z+spacing(z)))
    op = Operator([eq1,eq2], name="op")
    apply(op)
    @test isapprox(data(g), data(h))
end

@testset "shift_localindicies" begin
    @test Devito.shift_localindicies(1,2) == 0
end

@testset "AbstractDimension" begin
    x = SpaceDimension(name="x")
    @show typeof(x)
    @show typeof(x) <: Devito.AbstractDimension
    @test PyObject(x) == x.o
end

nothing
