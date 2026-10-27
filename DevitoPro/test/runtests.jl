using Devito, MPI

@info """
MPI implementation is set via test/LocalPreferences.toml (MPIPreferences.binary=\$(MPIPreferences.binary)).
CondaPkg non-defaults for the test subprocess must be set via environment variables, e.g.:

    JULIA_CONDAPKG_BACKEND=MicroMamba JULIA_CONDAPKG_PIP_BACKEND=pip julia -e 'using Pkg; Pkg.test("DevitoPro")'
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
