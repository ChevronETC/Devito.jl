using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
try
    Conda.add("pip")
    pip = joinpath(Conda.BINDIR, "pip")
    run(`$pip install cython`) 
    run(`$pip install versioneer`) 
    if dpro_repo != ""
        run(`$pip install git+$(dpro_repo)`)
    else
        run(`$pip install git+https://github.com/devitocodes/devito.git`)
    end
    run(`$pip install pytest`)
    run(`$pip install scipy`)
    run(`$pip install mpi4py`)
    run(`$pip install ipyparallel`)
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn unable to build
    else
        throw(e)
    end
end
