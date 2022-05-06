for testscript in ("serialtests.jl", "gencodetests.jl")
    include(testscript)
end

run(`mpirun -n 2 julia mpitests_2ranks.jl`)
run(`mpirun -n 4 julia mpitests_4ranks.jl`)
