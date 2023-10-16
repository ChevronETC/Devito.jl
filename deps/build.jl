using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
which_devito = get(ENV,"DEVITO_BRANCH", "master")
try
    Conda.pip_interop(true)
    @info "Building devito from branch $(which_devito)"
    Conda.pip("install", "devito[tests,extras,mpi]@git+https://github.com/devitocodes/devito@$(which_devito)")
    # optional devito pro installation
    if dpro_repo != ""
        Conda.pip("install","git+$(dpro_repo)")
    end

catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn "unable to build"
    else
        throw(e)
    end
end
