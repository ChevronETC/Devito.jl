using Devito, LinearAlgebra, Random, PyCall, Strided, Test

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", false)

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
    @test extent(grid) == ex
    @test origin(grid) == ori
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

    p = Constant(name="p", dtype=Float64, value=π)
    @test typeof(value(p)) == Float64
    @test value(p) == convert(Float64,π)
    @test data(p) == value(p)
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

@testset "Sparse time function coordinates, n=$n" for n in ( (10,11), (10,11,12) )
    grid = Grid(shape=n, dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
    stf_coords = coordinates(stf)
    @test isa(stf_coords, Devito.DevitoArray)
    @test size(stf_coords) == (length(n),10)
    x = rand(length(n),10)
    stf_coords .= x
    _stf_coords = coordinates(stf)
    @test _stf_coords ≈ x
end

@testset "Sparse time function coordinates, 3D grid" begin
    grid = Grid(shape=(10,11,12), dtype=Float32)
    stf = SparseTimeFunction(name="stf", npoint=10, nt=100, grid=grid)
    stf_coords = coordinates(stf)
    @test isa(stf_coords, Devito.DevitoArray)
    @test size(stf_coords) == (3,10)
    x = rand(3,10)
    stf_coords .= x
    _stf_coords = coordinates(stf)
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
    subdoms = (SubDomain("subdom0",[("left",2),("middle",0,0)]),SubDomain("subdom1",(("left",2),("middle",0,0))),SubDomain("subdom2",("left",2),("middle",0,0)),SubDomain("subdom3",[("left",2),(nothing,)]))
    for dom in subdoms
        grid = Grid(shape=(11,11), dtype=Float32, subdomains=dom)
        f = Devito.Function(name="f", grid=grid)
        d = data(f)
        d .= 1.0
        op = Operator([Eq(f,2.0,subdomain=dom)],name="write"*name(dom))
        apply(op)
        data(f)
        @test data(f)[1,5] == 2.
        @test data(f)[end,5] == 1.
        # get dimensions, reverse subdomain dimsbecause python object was returned
        griddim = dimensions(grid)
        subdim  = dimensions(subdomains(grid)[name(dom)])
        # test that a new subdimension created on first axis
        @test griddim[1] != subdim[1]
        # and that it is is derived 
        @test is_Derived(subdim[1]) == true 
        # and that the griddim not derived 
        @test is_Derived(griddim[1]) == false 
        # test that the complete second axis is same dimension
        @test griddim[2] == subdim[2]
        # test the interior method
        @test interior(grid) == subdomains(grid)["interior"]

    end
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
    g = Devito.Function(name="g", grid=grid)
    dg = data(g)
    dg .= 1.0
    h = Devito.Function(name="h", grid=grid)
    dh = data(h)
    dh .= 2.0
    k = Devito.Function(name="k", grid=grid)
    dh = data(h)
    dh .= 3.0
    op = Operator([Eq(mn,Min(g,f,h,k,4)),Eq(mx,Max(g,f,h,k,4))],name="minmax")
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
                @test data(g)[i] ≈ Base.$F(vals[i])
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
                @test data(g)[i] ≈ Base.$B(vals[i])
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
                @test data(g)[i] ≈ Base.$F(vals[i])
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
                @test data(g)[i] ≈ Base.$B(vals[i])
            end
        end
    end  
end

@testset "Unitary Minus" begin
    grid = Grid(shape=(11,), dtype=Float32)
    f = Devito.Function(name="f", grid=grid)
    g = Devito.Function(name="g", grid=grid)
    x = dimensions(f)[1]
    data(f) .= 1.0
    op = Operator([Eq(f,-f),Eq(g,-x)],name="unitaryminus")
    apply(op)
    for i in 1:length(data(f))
        @test data(f)[i] == -1.0
        @test data(g)[i] == 1-i
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
    @test 2*x+y-x == x+y
    @test x*x == x^2
    @test x+x+a == 2*x+a
    @test 2*x+a-x == x+a
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
    @test factor(f) == 2
    for (dim,attribute) in ((_dim,_attribute) for _dim in (a,b,c,d,e,f) for _attribute in attribtes)
        @eval begin
            @test $attribute($dim) == $dim.o.$attribute
        end
    end
end

@testset "Devito SubDimensions" begin
    d = SpaceDimension(name="d")
    dl = SubDimensionLeft(name="dl", parent=d, thickness=2)
    dr = SubDimensionRight(name="dr", parent=d, thickness=3)
    dm = SubDimensionMiddle(name="dm", parent=d, thickness_left=2, thickness_right=3)
    for subdim in (dl,dr,dm)
        @test parent(subdim) == d
    end
    @test (thickness(dl)[1][2], thickness(dl)[2][2]) == (2, 0)
    @test (thickness(dr)[1][2], thickness(dr)[2][2]) == (0, 3)
    @test (thickness(dm)[1][2], thickness(dr)[2][2]) == (2, 3)
end

@testset "Devito stepping dimension" begin
    grid = Grid(shape=(5,5),origin=(0.,0.),extent=(1.,1.))
    f = TimeFunction(grid=grid,space_order=8,time_order=2,name="f")
    @test stepping_dim(grid) == time_dim(f)
    @test stepping_dim(grid) != time_dim(grid)
    @test stepping_dim(grid).o.is_Stepping
end

@testset "Sparse Inject and Interpolate" begin
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
    coords =  [0.5; 0]
    src_coords = coordinates(src)
    src_coords .= coords
    src_data = data(src)
    src_data .= reshape(1e3*Base.sin.(time_range .* (3*pi/2)),:,1)
    pupdate = Eq(forward(p),1+p)
    src_term = inject(src; field=forward(p), expr=src*spacing(t)^2)

    rec = SparseTimeFunction(name="rec", grid=grid, npoint=2, nt=nt)
    rec_coords = coordinates(rec)
    rec_coords[:,1] .= coords
    rec_coords[:,2] .= reverse(coords)
    rec_term = interpolate(rec, expr=p)

    op = Operator([pupdate,src_term,rec_term],name="SparseInjectInterp", subs=smap)
    apply(op,time_M=nt-1)
    @test data(p)[3,1,end-1] ≈ (nt-1) + sum(src_data[1:end-1])*dt^2
    @test data(rec)[end,1] ≈ (nt-1) + sum(src_data[1:end-1])*dt^2
    @test data(rec)[end,2] ≈ (nt-1)
    
end

@testset "Left and Right Derivatives" begin
    grid = Grid(shape=(5),origin=(0.),extent=(1.))
    f = Devito.Function(grid=grid,name="f")
    g1 = Devito.Function(grid=grid,name="g1")
    g2 = Devito.Function(grid=grid,name="g2")
    data(f)[3] = 1.
    eq1 = Eq(g1,dxl(f))
    eq2 = Eq(g2,dxr(f))
    op = Operator([eq1,eq2],name="Derivatives")
    apply(op)
    @test data(g1)[2] == 0.
    @test data(g2)[2] == 4.
    @test data(g1)[3] == 4.
    @test data(g2)[3] == -4.
    @test data(g1)[4] == -4.
    @test data(g2)[4] == 0.
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
        @test dx(dx(x)+1) == 0
        @test dxl(dxl(x)+1) == 0
        @test dxr(dxr(x)+1) == 0
        @test dy(dy(x)+1) == 0
        @test dyl(dyl(x)+1) == 0
        @test dyr(dyr(x)+1) == 0
        @test dz(dz(x)+1) == 0
        @test dzl(dzl(x)+1) == 0
        @test dzr(dzr(x)+1) == 0
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
    @test factor(ci) == factr
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
    x = Devito.SpaceDimension(name="x")
    y = Devito.SpaceDimension(name="y")
    grd = Grid(shape=(5,5),dimensions=(y,x))
    f1 = Devito.Function(name="f1", grid=grd)
    f2 = Devito.Function(name="f2", grid=grd)
    f3 = Devito.Function(name="f3", grid=grd)
    g = Devito.Function(name="g", grid=grd)
    data(f1) .= 1.0
    data(f2) .= 1.0
    data(g)[:,3:end] .= 1.0
    ci1 = ConditionalDimension(name="ci1", parent=y, condition=And(Ne(g, 0), Lt(y, 2)))
    ci2 = ConditionalDimension(name="ci2", parent=y, condition=And(Ne(g, 0), Le(y, 2)))
    # Note, the order of dimensions feeding into parent seems to matter.  
    # If the grid dimensions were (x,y) 
    # the above conditional dimension would cause a compilation error on the operator at runtime.
    eq1 = Eq(f1, f1 + g, implicit_dims=ci1)
    eq2 = Eq(f2, f2 + g, implicit_dims=ci2)
    eq3 = Eq(f3, f2-f1)
    op = Operator([eq1,eq2,eq3],name="Implicit")
    apply(op)
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

@testset "Retrieve time_dim" begin
    g = Grid(shape=(5,5))
    @test time_dim(g) == dimension(g.o.time_dim)
    t = TimeDimension(name="t")
    f = TimeFunction(name="f",time_dim=t,grid=g)
    @test time_dim(f) == t
    @test time_dim(f) == dimensions(f)[end]
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
    @test nsimplify((2*x+2)/2) == x+1
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
    stf = SparseTimeFunction(name="stf", npoint=1, nt=10, grid=grid)
    @test name(stf) == "stf"
    op = Operator(Eq(f,1), name="op")
    @test name(op) == "op"
end

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
            @test evaluate(h) == g
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
