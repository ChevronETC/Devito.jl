using Devito

for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

# JKW: disabling mpi tests for now, we expect to remove MPI features from Devito.jl in future PR
# run(`$(mpiexec()) -n 2 julia --code-coverage mpitests_2ranks.jl`)
# run(`$(mpiexec()) -n 4 julia --code-coverage mpitests_4ranks.jl`)

if Devito.has_devitopro()
    @info "running devito pro tests"
    include("devitoprotests.jl")
else
    @info "not running devito pro tests"
end
