using Devito, MPI, MPIPreferences

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

# MPI.jl and mpi4py share this process, so they must load a single libmpi.
libmpis = filter(contains("libmpi.so"), Base.Libc.Libdl.dllist())
length(libmpis) == 1 || error("MPI.jl and mpi4py loaded different MPI libraries: $libmpis. Configure MPI.jl as described above.")

# Backend-agnostic suite. Also reused by DevitoPro's tests via include(), so the
# mpiexec script path is absolute (@__DIR__) to survive being included cross-package.
for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl", "csotests.jl")
    include(testscript)
end

@info "mpi tests with DEVITO_AUTOPADDING=0"
withenv("DEVITO_AUTOPADDING" => "0") do
    run(`$(mpiexec()) -n 2 julia --project=$(Base.active_project()) --code-coverage $(joinpath(@__DIR__, "mpitests_2ranks.jl"))`)
end
