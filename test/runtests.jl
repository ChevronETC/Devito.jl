for testscript in ("serialtests.jl", "gencodetests.jl")
    include(testscript)
end

run(`mpirun -n 2 julia mpitests.jl`)
