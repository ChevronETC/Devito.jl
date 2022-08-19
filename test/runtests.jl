for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

run(`mpirun -n 2 julia --code-coverage mpitests_2ranks.jl`)
run(`mpirun -n 4 julia --code-coverage mpitests_4ranks.jl`)
