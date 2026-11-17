using Devito, MPI

@info """
MPI.jl must load mpi4py's libmpi, the CondaPkg environment's Open MPI. Configure it once from the project
that runs Pkg.test (which needs a shared CondaPkg env, see below):
    using CondaPkg, MPIPreferences
    MPIPreferences.use_system_binary(; library_names=[CondaPkg.envdir("lib", "libmpi.so")], mpiexec=CondaPkg.envdir("bin", "mpiexec"))
CondaPkg settings come from test/LocalPreferences.toml plus the preferences of the project that runs
Pkg.test; set env = "@<name>" there to reuse one conda environment across test runs.
With a shared environment, wrap Pkg.test in withenv("JULIA_CONDAPKG_OFFLINE" => "yes") do ... end to skip
reinstalling the Python packages (and rebuilding devitopro) on every run; after changing Python
dependencies, run CondaPkg.resolve(force=true) from that project first.
"""

@assert Devito.has_devitopro() "DevitoPro test environment must provide devitopro"

# Reuse the shared open-source suite from DevitoOpen (included by file, not depended on
# as a package — depending on DevitoOpen would pull `devito` alongside `devitopro`).
include(joinpath(@__DIR__, "..", "..", "DevitoOpen", "test", "runtests.jl"))

# devitopro-only tests.
@info "running devito pro tests"
include(joinpath(@__DIR__, "devitoprotests.jl"))

@info "running pro tests with the decoupler"
withenv("DEVITO_DECOUPLER" => "1", "DEVITO_DECOUPLER_WORKERS" => "2", "MPI4PY_RC_RECV_MPROBE" => "0") do
    run(`$(mpiexec()) -n 1 julia --project=$(Base.active_project()) --code-coverage $(joinpath(@__DIR__, "devitoprotests.jl"))`)
end
