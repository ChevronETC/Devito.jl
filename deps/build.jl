using Conda

try
    Conda.add("pip")
    pip = joinpath(Conda.BINDIR, "pip")
    run(`$pip install cython`) 
    run(`$pip install versioneer`) 
    run(`$pip install devito`)
    run(`$pip install devito["mpi"]`)
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn unable to build
    else
        throw(e)
    end
end
