using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
try
    Conda.pip_interop(true)
    Conda.pip("install", "devito[tests,extras,mpi]@git+https://github.com/devitocodes/devito@master")
    # optional devito pro installation
    if dpro_repo != ""
        Conda.pip("install","git+$(dpro_repo)")
    end

catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn unable to build
    else
        throw(e)
    end
end
