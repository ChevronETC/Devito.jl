using Devito, Logging, PythonCall, Test

@testset "ABox Expanding Source" begin
    g = Grid(shape=(8,8), extent=(7.0,7.0))
    nt = 3
    coords = [0.5 2.5; 2.5 2.5; 0.5 4.5; 2.5 4.5]
    vp = Devito.Function(name="vp", grid=g, space_order=0)
    src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords)
    data(vp) .= 1.0
    abox = ABox(src, nothing, vp, -1)
    dt = 1.0
    srcbox_discrete = Devito.compute(abox; dt=dt)
    @show srcbox_discrete
    @test isapprox(pyconvert(Array, srcbox_discrete), [0 4 2 2; 0 3 1 1; 0 2 0 0])
    @test isequal(g, Devito.grid(abox))
    @test isequal(src, Devito.src(abox))
    @test isequal(nothing, Devito.rcv(abox))
    @test isequal(vp, Devito.vp(abox))
    @test pyconvert(Bool, eps(abox) == abox.o.eps)
end

@testset "ABox subdomains" begin
    mid = SubDomain("mid",[("middle",2,2),("middle",0,0)])
    g = Grid(shape=(8,8), extent=(7.0,7.0), subdomains=(mid,))
    nt = 3
    coords = [0.5 2.5; 2.5 2.5; 0.5 4.5; 2.5 4.5]
    vp = Devito.Function(name="vp", grid=g, space_order=0)
    src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords)
    data(vp) .= 1.0
    abox = ABox(src, nothing, vp, -1, subdomains=(mid,))
    dt = 1.0
    srcbox_discrete = Devito.compute(abox; dt=dt)
    sd = Devito.subdomains(abox)
    @test sd !== nothing
end

@testset "ABox errors without devitopro" begin
    g = Grid(shape=(8,8), extent=(7.0,7.0))
    nt = 3
    coords = [0.5 2.5; 2.5 2.5; 0.5 4.5; 2.5 4.5]
    vp = Devito.Function(name="vp", grid=g, space_order=0)
    src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords)
    data(vp) .= 1.0

    # unset devitopro by replacing with devito module
    PythonCall.pycopy!(Devito.devitopro, pyimport("devito"))
    @test_throws ErrorException ABox(src, nothing, vp, -1)
    
    # reset devitopro 
    try
        PythonCall.pycopy!(Devito.devitopro, pyimport("devitopro"))
    catch e
        PythonCall.pycopy!(Devito.devitopro, pyimport("devito"))
    end
end

# TODO (9/2/2025) - failing with decoupler, mloubout is looking into the issue
if get(ENV, "DEVITO_DECOUPLER", "0") != "1"
    # TODO - 2024-08-15 JKW these two ABox tests are broken -- some kind of API change?
    @testset "ABox Time Function" begin
        g = Grid(shape=(5,5), extent=(4.0,4.0))
        nt = 3
        coords = [2. 2. ;]
        space_order = 0
        vp = Devito.Function(name="vp", grid=g, space_order=space_order)
        src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords, space_order=0)
        data(vp) .= 1.0
        dt = 1.0
        t = time_dim(g)
        abox = ABox(src, nothing, vp, space_order)
        # This needs layers = nothing as currently setup to automatically
        # deffault to disk and buffered/saved functions cannot be used like that in an equation
        u = TimeFunction(name="u", grid=g, save=nt, space_order=space_order, layers=nothing)
        op = Operator([Eq(forward(u), t+1, subdomain=abox)])
        apply(op, dt=dt)
        @test data(u)[:,:,1] ≈ zeros(Float32, 5 , 5)
        @test data(u)[2:end-1,2:end-1,2] ≈ ones(Float32, 3, 3)
        data(u)[2:end-1,2:end-1,2] .= 0
        @test data(u)[:,:,2] ≈ zeros(Float32, 5 , 5)
        @test data(u)[:,:,3] ≈ 2 .* ones(Float32, 5 , 5)
    end
end

# TODO (9/2/2025)- failing with decoupler, mloubout is looking into the issue
if get(ENV, "DEVITO_DECOUPLER", "0") != "1"
    @testset "ABox Intersection Time Function" begin
        mid = SubDomain("mid",[("middle",2,2),("middle",0,0)])
        g = Grid(shape=(5,5), extent=(4.0,4.0), subdomains=mid)
        nt = 3
        coords = [2. 2. ;]
        space_order = 0
        vp = Devito.Function(name="vp", grid=g, space_order=space_order)
        src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords, space_order=0)
        data(vp) .= 1.0
        dt = 1.0
        t = time_dim(g)
        abox = ABox(src, nothing, vp, space_order)
        intbox = Devito.intersection(abox,mid)
        # Similar as above, need layers=nothing
        u = TimeFunction(name="u", grid=g, save=nt, space_order=space_order, layers=nothing)
        op = Operator([Eq(forward(u), t+1, subdomain=intbox)])
        apply(op, dt=dt)
        @test data(u)[:,:,1] ≈ zeros(Float32, 5 , 5)
        @test data(u)[3,2:4,2] ≈ ones(Float32, 3)
        data(u)[3,2:4,2] .= 0
        @test data(u)[:,:,2] ≈ zeros(Float32, 5 , 5)
        @test data(u)[3,:,3] ≈ 2 .* ones(Float32, 5)
        data(u)[3,:,3] .= 0
        @test data(u)[:,:,3] ≈ zeros(Float32, 5 , 5)
    end
end

@testset "FloatX dtypes with $(mytype), $(DT), $(CT)" for mytype ∈ [Float32, Float64], (nb, DT, CT) in zip([8, 16], [FloatX8, FloatX16], [UInt8, UInt16])
    g = Grid(shape=(5,5))
    dtype = DT(1.5f0, 4.5f0)
    atol = Devito.scale(dtype)

    @test Devito.compressed_type(dtype) == CT
    @test Devito.compressed_type(dtype(1)) == CT

    # Scale and offset
    @test Devito.nbytes(dtype) == nb
    @test Devito.nbytes(dtype(1)) == nb

    @test Devito.offset(dtype) == 1.5f0
    @test Devito.offset(dtype(1)) == 1.5f0

    @test Devito.scale(dtype) == (4.5f0 - 1.5f0) / (2^nb - 1)
    @test Devito.scale(dtype(1)) == (4.5f0 - 1.5f0) / (2^nb - 1)

    @test dtype(1.5f0).value == CT(0)
    @test dtype(4.5f0) == CT(2^nb - 1)

    # Arrays. zeros is defined as initializer eventhough out of range
    # to avoid initialization issues
    a = zeros(dtype, 5, 5)
    @test a[1, 1].value == CT(0)
    a = ones(dtype, 5, 5)
    @test a[1, 1].value == CT(1)
    a .= 1.5f0
    @test all(isapprox.(a, 1.5f0; rtol=0, atol=atol))
    a .= 1f0 .+ 3f0
    @test all(isapprox.(a, 4f0; rtol=0, atol=atol))
    a .= 4.5f0
    @test all(a .== 4.5f0)
    a .= a .- a ./ 2f0
    @test all(isapprox.(a, 2.25f0; rtol=0, atol=atol))

    # Now test function
    f = Devito.Function(name="f", grid=g, dtype=dtype, space_order=0)
    @test eltype(data(f)) == dtype
    data(f) .= 1.5f0
    @test all(data(f) .== 1.5f0)
end

@testset "FloatX addition DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(1.5f0)
    b = dtype(1.5f0)
    @test Base.:+(a,b) ≈ dtype(1.5f0 + 1.5f0).value
    @test Base.:+(1.5f0,b) ≈ dtype(1.5f0 + 1.5f0).value
    @test Base.:+(a,1.5f0) ≈ dtype(1.5f0 + 1.5f0).value
end

@testset "FloatX subtraction DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(3.0f0)
    b = dtype(1.5f0)
    @test Base.:-(a,b) ≈ dtype(3.0f0 - 1.5f0).value
    @test Base.:-(3.0f0,b) ≈ dtype(3.0f0 - 1.5f0).value
    @test Base.:-(a,1.5f0) ≈ dtype(3.0f0 - 1.5f0).value
end

@testset "FloatX multiplication DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(1.5f0)
    b = dtype(1.5f0)
    @test Base.:*(a,b) ≈ dtype(1.5f0 * 1.5f0).value
    @test Base.:*(1.5f0,b) ≈ dtype(1.5f0 * 1.5f0).value
    @test Base.:*(a,1.5f0) ≈ dtype(1.5f0 * 1.5f0).value
end

@testset "FloatX division DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(3.0f0)
    b = dtype(1.5f0)
    @test Base.:/(a,b) ≈ dtype(3.0f0 / 1.5f0).value
    @test Base.:/(3.0f0,b) ≈ dtype(3.0f0 / 1.5f0).value
    @test Base.:/(a,1.5f0) ≈ dtype(3.0f0 / 1.5f0).value
end

@testset "FloatX comparison DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(1.5f0)
    b = dtype(1.5f0)
    @test a == b
    @test a == 1.5f0
    @test 1.5f0 == b
    @test Base.isapprox(a,b)
    @test Base.isapprox(1.5f0,b)
    @test Base.isapprox(a,1.5f0)
end

@testset "FloatX convert DT=$(DT)" for DT ∈ (FloatX8, FloatX16)
    dtype = DT(1.5f0, 4.5f0)
    a = dtype(1.5f0)
    @test Base.convert(typeof(a),1.5f0) == a
    @test Base.convert(Float32,a) ≈ 1.5f0
end

@testset "FloatX promote_rule tests" begin
    fmin,fmax = 1.5, 4.5
    f32u08 = Devito.FloatX{fmin,fmax,Float32,UInt8}(Float32(2))
    f32u16 = Devito.FloatX{fmin,fmax,Float32,UInt16}(Float32(2))
    f64u08 = Devito.FloatX{fmin,fmax,Float64,UInt8}(Float64(2))
    f64u16 = Devito.FloatX{fmin,fmax,Float64,UInt16}(Float64(2))
    @test promote_type(typeof(f32u08), typeof(f32u16)) == typeof(f32u16)
    @test promote_type(typeof(f64u08), typeof(f64u16)) == typeof(f64u16)
    @test promote_type(typeof(f32u08), typeof(f64u08)) == typeof(f64u08)
    @test promote_type(typeof(f32u08), typeof(f64u16)) == typeof(f64u16)
end

@testset "FloatX arrays with $(mytype), $(DT), $(CT), autopad=$(autopad)" for mytype ∈ [Float32, Float64], (DT, CT) in zip([FloatX8, FloatX16], [UInt8, UInt16]), autopad ∈ (true,false)
    configuration!("autopadding", autopad)
    g = Grid(shape=(5,5))
    dtype = DT(mytype(-1.1), mytype(+1.1))
    f = Devito.Function(name="f", grid=g, dtype=dtype, space_order=8)
    g = Devito.Function(name="g", grid=g, dtype=dtype, space_order=8)
    values = -1 .+ 2 .* rand(mytype,size(g))
    data(f) .= values
    copyto!(data(g),values)
    @test isapprox(Devito.decompress.(data(f)), Devito.decompress.(data(g)))
end

@testset "FloatX eps with $(mytype), $(DT), $(CT)" for mytype ∈ [Float32, Float64], (DT, CT) in zip([FloatX8, FloatX16], [UInt8, UInt16])
    g = Grid(shape=(5,5))
    dtype = DT(mytype(1.5), mytype(4.5))
    @test eps(dtype) ≈ eps(mytype)
end

devito_arch = get(ENV, "DEVITO_ARCH", "gcc")

#Ashish - commenting out for now
# # TODO (9/2/2025) - failing with decoupler, mloubout is looking into the issue
# if get(ENV, "DEVITO_DECOUPLER", "0") != "1"
#     @testset "CCall with printf" begin
#         carch = devito_arch in ["gcc", "clang"] ? devito_arch : "gcc"
#         switchconfig_cm = Devito.switchconfig(;compiler=get(ENV, "CC", carch))
        
#         switchconfig_cm.__enter__()
#         try
#             pf = CCall("printf", header="stdio.h")
#             @test Devito.name(pf) == "printf"
#             @test Devito.header(pf) == "stdio.h"
#             @test Devito.header_dirs(pf) == pf.o.header_dirs
#             @test Devito.libs(pf) == pf.o.libs
#             @test Devito.lib_dirs(pf) == pf.o.lib_dirs
#             @test Devito.target(pf) == pf.o.target
#             @test Devito.types(pf) == pf.o.types
#             @test Py(pf) == pf.o
#             printingop = Operator([pf([""" "hello world!" """])])
#             ccode(printingop, filename="helloworld.c")
#             code = read("helloworld.c", String)
#             @test occursin("#include \"stdio.h\"\n", code)
#             @test occursin("printf( \"hello world!\" );\n", code)
#             @test try apply(printingop)
#                 true
#             catch
#                 false
#             end
#             rm("helloworld.c", force=true)
#         finally
#             switchconfig_cm.__exit__(pybuiltins.None, pybuiltins.None, pybuiltins.None)
#         end
#     end

#     @testset "CCall errors without devitopro" begin
#         # unset devitopro
#         PythonCall.pycopy!(Devito.devitopro, pyimport("devito"))
#         @test_throws ErrorException CCall("printf", header="stdio.h")
#         # reset devitopro 
#         try
#             PythonCall.pycopy!(Devito.devitopro, pyimport("devitopro"))
#         catch e
#             PythonCall.pycopy!(Devito.devitopro, pyimport("devito"))
#         end
#     end
# end

# # currently only gcc and nvc are useful
# compression = []
# (lowercase(devito_arch) == "nvc") && (push!(compression, "bitcomp"))
# (lowercase(devito_arch) in ["gcc", "clang"]) && (push!(compression, "cvxcompress"))

# @testset "Serialization with compression=$(compression)" for compression in compression
#     if compression == "bitcomp"
#         configuration!("compiler", "nvc")
#     else
#         configuration!("compiler", devito_arch)
#     end

#     nt = 11
#     space_order = 8
#     grid = Grid(shape=(21,21,21), dtype=Float32)
#     f1 = TimeFunction(name="f1", grid=grid, space_order=space_order, time_order=1, save=Buffer(1))
#     f2 = TimeFunction(name="f2", grid=grid, space_order=space_order, time_order=1, save=Buffer(1))
#     z, y, x, t = dimensions(f1)
#     ct = ConditionalDimension(name="ct", parent=time_dim(grid), factor=1)
#     dumpdir = joinpath(tempdir(),"test-bitcomp")
#     isdir(dumpdir) && rm(dumpdir, force=true, recursive=true)
#     mkdir(dumpdir)
#     flazy = TimeFunction(name="flazy", lazy=false, grid=grid, time_order=0, space_order=space_order, time_dim=ct, save=nt, compression=compression, serialization=dumpdir)

#     eq1 = Eq(forward(f1),f1+1)
#     eq2 = Eq(flazy, f1)
#     if compression == "bitcomp"
#         op1 = Operator([eq1,eq2], name="OpTestBitcompCompress", nbits=24)
#         apply(op1, time_m=1, time_M=nt-1)
#     else
#         op1 = Operator([eq1,eq2], name="OpTestCvxCompress")
#         apply(op1, time_m=1, time_M=nt-1, compscale=1.0e-6)
#     end
    
#     eq3 = Eq(f2, flazy)
#     op2 = Operator([eq3], name="OpTestDecompress")
#     for kt = 1:nt-1
#         if compression == "bitcomp"
#             apply(op2, time_m=1, time_M=kt)
#         else
#             apply(op2, time_m=1, time_M=kt)
#         end
#         @show kt,extrema(data(f2)[:,:,:,1])
#         @test minimum(data(f2)) ≈ Float32(kt)
#         @test maximum(data(f2)) ≈ Float32(kt)
#     end
# end

@testset "Serialization serial2str" begin
    nt = 11
    space_order = 8
    grid = Grid(shape=(21,21,21), dtype=Float32)
    f1 = TimeFunction(name="f1", grid=grid, space_order=space_order, time_order=1, save=Buffer(1))
    f2 = TimeFunction(name="f2", grid=grid, space_order=space_order, time_order=1, save=Buffer(1))
    z, y, x, t = dimensions(f1)
    ct = ConditionalDimension(name="ct", parent=time_dim(grid), factor=1)
    dumpdir = joinpath(tempdir(),"test-serial2str")
    isdir(dumpdir) && rm(dumpdir, force=true, recursive=true)
    mkdir(dumpdir)
    flazy = TimeFunction(name="flazy", lazy=false, grid=grid, time_order=0, space_order=space_order, time_dim=ct, save=nt, serialization=dumpdir)
    str = Devito.serial2str(flazy)
    ser = Devito.str2serial(str)
    pathlib = pyimport("pathlib")
    py_path = pathlib.Path(str)
end

@testset "TimeFunction, lazy streaming n=$n" for n in ( (4,5), (4,5,6) )
    g = Grid(shape=n)
    ff = Devito.Function(name="ff", grid=g, space_order=4)
    u1 = TimeFunction(name="u1", grid=g, space_order=4, lazy=false, allowpro=true, time_order=1, save=10, serialization="/tmp/u1", compression=nothing)
    u2 = TimeFunction(name="u2", grid=g, space_order=4, lazy=true,  allowpro=true, time_order=1, save=10, serialization="/tmp/u2", compression=nothing)
    u3 = TimeFunction(name="u2", grid=g, space_order=4, allowpro=false, time_order=1, save=10)
    @test u1 !== nothing
    @test u2 !== nothing
    @test u3 !== nothing
    @test_throws ErrorException TimeFunction(ff.o)
    @test_logs (:warn, "Object doesn't have serialized path!") Devito.serial2str(u3)
end

# JKW: removing for now, not sure what is even being tested here
# @testset "Serialization with CCall T=$T" for T in (Float32,Float64)
#     space_order = 2
#     time_M = 3
#     filename = "testserialization.bin"
#     fo = CCall("fopen", header="stdio.h")
#     fw = CCall("fwrite", header="stdio.h")
#     fc = CCall("fclose", header="stdio.h")
#     grid = Grid(shape=(4, 3), dtype=T)
#     time = time_dim(grid)
#     t = stepping_dim(grid)
    
#     stream = Pointer(name="stream") # pointer to file object
#     elesize = (T == Float32 ? 4 : 8)
#     u = TimeFunction(name="u", grid=grid, space_order=space_order)
#     @show size_with_halo(u)
#     @show prod(size_with_halo(u)[1:end-1])
#     nele = prod(size_with_halo(u)[1:end-1])
#     eqnswrite = [   fo([""" "$filename" """,""" "w" """], stream),
#         Eq(forward(u), u + 1.),
#         fw([Byref(indexed(u)[-space_order+1, -space_order+1, t+1]), elesize, nele, stream], implicit_dims=(time,)),
#         fc([stream])
#     ]
#     # CCall test written to use gcc
#     opwrite = Operator(eqnswrite, compiler="gcc")
#     apply(opwrite,time_M=time_M)

#     holder = zeros(T, size_with_halo(u)[1:end-1]..., time_M)
#     read!(filename, holder)
#     for it in 1:time_M
#         @test holder[space_order+1:end-space_order, space_order+1:end-space_order, it] ≈ it .* ones(T, size(u)[1:end-1]...)
#     end
#     rm(filename, force=true)
# end
