using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
which_devito = get(ENV,"DEVITO_BRANCH", "")
try
    Conda.pip_interop(true)
    # optional devito pro installation
    if dpro_repo != ""
        Conda.pip("install", "git+$(dpro_repo)")
        # Currently separate as very platform dependent
        Conda.pip("install", "mpi4py")
    elseif which_devito != ""
        @info "Building devito from branch $(which_devito)"
        Conda.pip("install", "devito[tests,extras,mpi]@git+https://github.com/devitocodes/devito@$(which_devito)")
    else
        @info "Building devito from latest release"
        Conda.pip("install", "devito[tests,extras,mpi]")
    end
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn "unable to build"
    else
        throw(e)
    end
end
