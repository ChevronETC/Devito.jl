using Devito, PyCall, Test

configuration!("log-level", "DEBUG")
configuration!("language", "openmp")
configuration!("mpi", false)

# test independent derivatives
function python_test_individual_derivatives() 
    python_code = 
        py"""
        import numpy as np
        from numpy.testing import assert_almost_equal
        from devito import Grid, Function, Eq, Operator

        nx,ny,nz = 11,11,11
        dx,dy,dz = 10,10,10
        grid = Grid(extent=(dx*(nx-1),dy*(ny-1),dz*(nz-1)), shape=(nx,ny,nz), origin=(0,0,0), dtype=np.float32)
        x,y,z = grid.dimensions

        fx = Function(name='fx', grid=grid, space_order=2)
        fy = Function(name='fy', grid=grid, space_order=2)
        fz = Function(name='fz', grid=grid, space_order=2)
        gx = Function(name='gx', grid=grid, space_order=2)
        gy = Function(name='gy', grid=grid, space_order=2)
        gz = Function(name='gz', grid=grid, space_order=2)

        a,b,c = 2,3,4

        eq_x0 = Eq(fx, x.spacing * a * x)
        eq_y0 = Eq(fy, y.spacing * b * y)
        eq_z0 = Eq(fz, z.spacing * c * z)

        eq_dx = Eq(gx, fx.dx(x0 = x + x.spacing / 2))
        eq_dy = Eq(gy, fy.dy(x0 = y + y.spacing / 2))
        eq_dz = Eq(gz, fz.dz(x0 = z + z.spacing / 2))

        spacing_map = grid.spacing_map
        op = Operator([eq_x0, eq_y0, eq_z0, eq_dx, eq_dy, eq_dz], subs=spacing_map, name="Op1")
        op.apply()

        print(gx.data[5,5,5])
        print(gy.data[5,5,5])
        print(gz.data[5,5,5])

        assert_almost_equal(a, gx.data[nx//2,ny//2,nz//2], decimal=6)
        assert_almost_equal(b, gy.data[nx//2,ny//2,nz//2], decimal=6)
        assert_almost_equal(c, gz.data[nx//2,ny//2,nz//2], decimal=6)

        f = open("operator1.python.c", "w")
        print(op, file=f)
        f.close()
        """
end

# test folding two discretiations in a mixed derivative
function python_test_mixed_derivatives() 
    python_code = 
        py"""
        import numpy as np
        from numpy.testing import assert_almost_equal
        from devito import Grid, Function, Eq, Operator

        nx,ny,nz = 11,11,11
        dx,dy,dz = 10,10,10
        grid = Grid(extent=(dx*(nx-1),dy*(ny-1),dz*(nz-1)), shape=(nx,ny,nz), origin=(0,0,0), dtype=np.float32)
        x,y,z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=2)
        g = Function(name='g', grid=grid, space_order=2)

        a,b,c = 2,3,4

        eq = Eq(g, f.dx.dy.dz)

        spacing_map = grid.spacing_map
        op = Operator([eq], subs=spacing_map, name="Op2")
        op.apply()

        print(gx.data[5,5,5])
        print(gy.data[5,5,5])
        print(gz.data[5,5,5])

        assert_almost_equal(a, gx.data[nx//2,ny//2,nz//2], decimal=6)
        assert_almost_equal(b, gy.data[nx//2,ny//2,nz//2], decimal=6)
        assert_almost_equal(c, gz.data[nx//2,ny//2,nz//2], decimal=6)

        f = open("operator2.python.c", "w")
        print(op, file=f)
        f.close()
        """
end

@testset "GenCodeDerivativesIndividual" begin

    # python execution
    python_test_individual_derivatives()

    # julia with Devito.jl executionl
    nz,ny,nx = 11,11,11
    dz,dy,dx = 10,10,10
    grid = Grid(extent=(dx*(nx-1),dy*(ny-1),dz*(nz-1)), shape=(nx,ny,nz), origin=(0,0,0), dtype=Float32)
    z,y,x = dimensions(grid)

    fz = Devito.Function(name="fz", grid=grid, space_order=2)
    fy = Devito.Function(name="fy", grid=grid, space_order=2)
    fx = Devito.Function(name="fx", grid=grid, space_order=2)

    gz = Devito.Function(name="gz", grid=grid, space_order=2)
    gy = Devito.Function(name="gy", grid=grid, space_order=2)
    gx = Devito.Function(name="gx", grid=grid, space_order=2)

    a,b,c = 2,3,4

    eq_z0 = Eq(fz, spacing(z) * c * z)
    eq_y0 = Eq(fy, spacing(y) * b * y)
    eq_x0 = Eq(fx, spacing(x) * a * x)

    eq_dz = Eq(gz, Devito.dz(fz,x0=z+spacing(z)/2))
    eq_dy = Eq(gy, Devito.dy(fy,x0=y+spacing(y)/2))
    eq_dx = Eq(gx, Devito.dx(fx,x0=x+spacing(x)/2))

    spacing_map = Devito.spacing_map(grid)
    op = Operator([eq_x0, eq_y0, eq_z0, eq_dx, eq_dy, eq_dz], subs=spacing_map, name="Op1")
    apply(op)

    ccode(op; filename="operator1.julia.c")

    @show data(gx)[5,5,5]
    @show data(gy)[5,5,5]
    @show data(gz)[5,5,5]

    @test data(gx)[5,5,5] == a
    @test data(gy)[5,5,5] == b
    @test data(gz)[5,5,5] == c

    # check parity of generated code
    @test success(`cmp --quiet operator1.julia.c operator1.python.c`)

end


@testset "GenCodeDerivativesMixed" begin

    # python execution
    python_test_mixed_derivatives()

    # julia with Devito.jl executionl
    nz,ny,nx = 11,11,11
    dz,dy,dx = 10,10,10
    grid = Grid(extent=(dx*(nx-1),dy*(ny-1),dz*(nz-1)), shape=(nx,ny,nz), origin=(0,0,0), dtype=Float32)
    z,y,x = dimensions(grid)

    f = Devito.Function(name="f", grid=grid, space_order=2)
    g = Devito.Function(name="g", grid=grid, space_order=2)

    a,b,c = 2,3,4

    eq = Eq(g, Devito.dz(Devito.dy(Devito.dx(f))))

    spacing_map = Devito.spacing_map(grid)
    op = Operator([eq], subs=spacing_map, name="Op2")
    apply(op)

    ccode(op; filename="operator2.julia.c")

    # check parity of generated code
    @test success(`cmp --quiet operator2.julia.c operator2.python.c`)

end
