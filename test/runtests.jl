using Devito, MPI, MPIPreferences

@info """
When running via the `Pkg.test` method, the MPI implementation is set via the test/LocalPreferences.toml file.
MPIPreferences.binary=$(MPIPreferences.binary).

To change to a different implementation do (for example):

    cd(DEPOT_PATH[1] * "/dev/Devito)
    ]activate .
    using MPIPreferences
    MPIPreferences.use_jll_binary("MPICH_jll")
"""

for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

# Only run devitopro tests if devitopro is available
if Devito.has_devitopro()
    @info "running devito pro tests"

    include("devitoprotests.jl")
    @info "running pro tests with the decoupler"
    withenv("DEVITO_DECOUPLER"=>"1", "DEVITO_DECOUPLER_WORKERS"=>"2", "MPI4PY_RC_RECV_MPROBE"=>"0") do
        run(`$(mpiexec()) -n 1 julia --code-coverage devitoprotests.jl`)
    end
else
    @info "not running devito pro tests"
end


@info "mpi tests with DEVITO_AUTOPADDING=0"
withenv("DEVITO_AUTOPADDING" => "0") do
    run(`$(mpiexec()) -n 2 julia --code-coverage mpitests_2ranks.jl`)
    run(`$(mpiexec()) -n 4 julia --code-coverage mpitests_4ranks.jl`)
end
