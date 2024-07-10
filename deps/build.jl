using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
which_devito = get(ENV,"DEVITO_BRANCH", "")
try

    Conda.pip_interop(true)
    # optional devito pro installation
    # note that the submodules do not install correctly from a URL, so we need install from local cloned repo
    if dpro_repo != ""
        @info "Building devitopro from latest release"
        Conda.pip("uninstall -y", "devitopro")
        Conda.pip("uninstall -y", "devito")
        
        dir = "$(tempname())-devitopro"
        _pwd = pwd()
        Sys.which("git") === nothing && error("git is not installed")
        run(`git clone $(dpro_repo) $(dir)`)
        cd(dir)
        run(`git submodule update --init`)
        cd(_pwd)

        Conda.pip("install", "$(dir)")
        rm(dir, recursive=true, force=true)

        ENV["CFLAGS"] = "-noswitcherror"
        Conda.pip("uninstall -y", "mpi4py")
        Conda.pip("install", "mpi4py")

    elseif which_devito != ""
        @info "Building devito from branch $(which_devito)"
        Conda.pip("install", "devito[tests,extras,mpi]@git+https://github.com/devitocodes/devito@$(which_devito)")

    else
        @info "Building devito from latest release"
        Conda.pip("uninstall -y", "devito")
        
        dir = "$(tempname())-devito"
        Sys.which("git") === nothing && error("git is not installed")
        run(`git clone https://github.com/devitocodes/devito $(dir)`)
        
        Conda.pip("install", "$(dir)[tests,extras,mpi]")
        rm(dir, recursive=true, force=true)
        
        Conda.pip("uninstall -y", "mpi4py")
        Conda.pip("install", "mpi4py")
        
        # Conda.pip("install", "devito[tests,extras,mpi]")
    end
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn "unable to build"
    else
        throw(e)
    end
end
