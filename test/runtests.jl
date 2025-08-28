using Devito, MPI

for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

# mloubout: Only run devitopro tests if devitopro is available
if Devito.has_devitopro()
    @info "running devito pro tests"
    include("devitoprotests.jl")
    # This should not trigger an "using MPI" and only rely on mpirun to trigger the decoupler inside devito
    @info "running pro tests with the decoupler"
    run(`env DEVITO_DECOUPLER=1 DEVITO_DECOUPLER_WORKERS=2 $(mpiexec()) -n 1 julia --code-coverage devitoprotests.jl`)
else
    @info "not running devito pro tests"
end

# mloubout: include mpi tests through extension
include("mpitests.jl")
