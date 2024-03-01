using Devito, MPI

for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

run(`$(mpiexec()) -n 2 julia --code-coverage mpitests_2ranks.jl`)
run(`$(mpiexec()) -n 4 julia --code-coverage mpitests_4ranks.jl`)

if Devito.has_devitopro()
    include("devitoprotests.jl")
end
