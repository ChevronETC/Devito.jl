using Devito, MPI, MPIPreferences

@info """
MPI implementation is set via test/LocalPreferences.toml (MPIPreferences.binary=$(MPIPreferences.binary)).
CondaPkg non-defaults for the test subprocess must be set via environment variables, e.g.:

    JULIA_CONDAPKG_BACKEND=MicroMamba JULIA_CONDAPKG_PIP_BACKEND=pip julia -e 'using Pkg; Pkg.test("DevitoOpen")'
"""

# Backend-agnostic suite. Also reused by DevitoPro's tests via include(), so the
# mpiexec script path is absolute (@__DIR__) to survive being included cross-package.
for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl", "csotests.jl")
    include(testscript)
end

@info "mpi tests with DEVITO_AUTOPADDING=0"
withenv("DEVITO_AUTOPADDING" => "0") do
    run(`$(mpiexec()) -n 2 julia --project=$(Base.active_project()) --code-coverage $(joinpath(@__DIR__, "mpitests_2ranks.jl"))`)
end
