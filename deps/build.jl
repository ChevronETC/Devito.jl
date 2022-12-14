using Conda

try
    Conda.add("pip")
    pip = joinpath(Conda.BINDIR, "pip")
    run(`$pip install cython`) 
    run(`$pip install versioneer`) 
    run(`$pip install git+https://github.com/devitocodes/devito.git`)
    run(`$pip install mpi4py`)
    run(`$pip install ipyparallel`)
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn unable to build
    else
        throw(e)
    end
end
# set the devito pro repo to DEVITO_PRO environmental variable
dpro_repo = get(ENV, "DEVITO_PRO", "")
if dpro_repo != ""
    @info "Building DevitoPro"
    try 
        pip = joinpath(Conda.BINDIR, "pip")
        run(`$pip install git+$(dpro_repo)`)
    catch e
        if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
            @warn unable to build
        else
            throw(e)
        end
    end
end

#run(`$pip install devito`)
#run(`$pip install devito[extras]`)
