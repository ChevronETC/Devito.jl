for testscript in (#="serialtests.jl",=#"mpitests.jl",)
    include(testscript)
end