using Devito, DevitoPro, Test

@testset "ABox Expanding Source" begin
    g = Grid(shape=(8,8), extent=(7.0,7.0))
    nt = 3
    coords = [0.5 2.5; 2.5 2.5; 0.5 4.5; 2.5 4.5]
    vp = Devito.Function(name="vp", grid=g, space_order=0)
    src = SparseTimeFunction(name="src",  grid=g, nt=nt, npoint=size(coords)[1], coordinates=coords)
    data(vp) .= 1.0
    abox = DevitoPro.ABox(src, nothing, vp, -1)
    dt = 1.0
    srcbox_discrete = DevitoPro.compute(abox; dt=dt)
    @test srcbox_discrete ≈ [0 4 2 2; 0 3 1 1; 0 2 0 0]
    @test isequal(g, DevitoPro.grid(abox))
    @test isequal(src, DevitoPro.src(abox))
    @test isequal(nothing, DevitoPro.rcv(abox))
    @test isequal(vp, DevitoPro.vp(abox))
end

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
    abox = DevitoPro.ABox(src, nothing, vp, space_order)
    u = TimeFunction(name="u", grid=g, save=nt, space_order=space_order)
    op = Operator([Eq(forward(u), t+1, subdomain=abox)])
    apply(op, dt=dt)
    @test data(u)[:,:,1] ≈ zeros(Float32, 5 , 5)
    @test data(u)[2:end-1,2:end-1,2] ≈ ones(Float32, 3, 3)
    data(u)[2:end-1,2:end-1,2] .= 0
    @test data(u)[:,:,2] ≈ zeros(Float32, 5 , 5)
    @test data(u)[:,:,3] ≈ 2 .* ones(Float32, 5 , 5)
end


@testset "CCall with printf" begin
    pf = CCall("printf", header="stdio.h")
    @test DevitoPro.name(pf) == "printf"
    @test DevitoPro.header(pf) == "stdio.h"
    printingop = Operator([pf([""" "hello world!" """])])
    ccode(printingop, filename="helloworld.c")
    # read the program
    code = read("helloworld.c", String)
    # check to make sure header is in the program
    @test occursin("#include \"stdio.h\"\n", code)
    # check to make sure the printf statement is in the program
    @test occursin("printf(\"hello world!\" );\n", code)
    # test to make sure the operator compiles and runs
    @test try apply(printingop)
        true
    catch
        false
    end
    # remove the file
    rm("helloworld.c", force=true)
end

@testset "Serialization with CCall T=$T" for T in (Float32,Float64)
    space_order = 2
    time_M = 3
    filename = "testserialization.bin"
    fo = CCall("fopen", header="stdio.h")
    fw = CCall("fwrite", header="stdio.h")
    fc = CCall("fclose", header="stdio.h")
    grid = Grid(shape=(4, 3), dtype=T)
    time = time_dim(grid)
    t = stepping_dim(grid)
    
    stream = Pointer(name="stream") # pointer to file object
    elesize = (T == Float32 ? 4 : 8)
    u = TimeFunction(name="u", grid=grid, space_order=space_order)
    nele = prod(size_with_halo(u)[1:end-1])
    eqnswrite = [   fo([""" "$filename" """,""" "w" """], stream),
                    Eq(forward(u), u + 1.),
                    fw([Byref(indexed(u)[-space_order+1, -space_order+1, t+1]), elesize, nele, stream], implicit_dims=(time,)),
                    fc([stream])
                ]
    opwrite = Operator(eqnswrite)
    apply(opwrite,time_M=time_M)

    holder = zeros(T, size_with_halo(u)[1:end-1]..., time_M)
    read!(filename, holder)
    for it in 1:time_M
        @test holder[space_order+1:end-space_order, space_order+1:end-space_order, it] ≈ it .* ones(T, size(u)[1:end-1]...)
    end
    rm(filename, force=true)
end
