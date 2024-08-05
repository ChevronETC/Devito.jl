using Conda

dpro_repo = get(ENV, "DEVITO_PRO", "")
which_devito = get(ENV,"DEVITO_BRANCH", "")
try

    Conda.pip_interop(true)
    # optional devito pro installation
    # note that the submodules do not install correctly from a URL, so we need install from local cloned repo
    # 2024-07-29 this is totally hacked for nvidia HPCX and Open MPI 4.1.7a1 
    # using nvidia HPC SDK 24.7 and cuda 12.5
    if dpro_repo != ""
        @info "Building devitopro from latest release"
        Conda.pip("uninstall -y", "devitopro")
        Conda.pip("uninstall -y", "devito")
        Conda.pip("uninstall -y", "numpy")
        
        # clone the devitopro repository and init submodules
        dir = "$(tempname())-devitopro"
        _pwd = pwd()
        Sys.which("git") === nothing && error("git is not installed")
        run(`git clone $(dpro_repo) $(dir)`)
        cd(dir)
        run(`git submodule update --init`)
        cd(_pwd)

        # use nvc compiler
        ENV["CC"] = "nvc"

        # devitopro
        Conda.pip("install", "$(dir)")
        rm(dir, recursive=true, force=true)

        # devito requirements
        Conda.pip("install --no-cache-dir -r", "https://raw.githubusercontent.com/devitocodes/devito/master/requirements.txt")

        # nvida requirements
        Conda.pip("install --no-cache-dir -r", "https://raw.githubusercontent.com/devitocodes/devito/master/requirements-nvidia.txt")

        # mpi requirements
        ENV["CFLAGS"] = "-noswitcherror -tp=px"
        Conda.pip("install --no-cache-dir -r", "https://raw.githubusercontent.com/devitocodes/devito/master/requirements-mpi.txt")
        delete!(ENV,"CFLAGS")

        # and finally ... mpi4py
        # ENV["CFLAGS"] = "-noswitcherror -tp=px"
        # Conda.pip("uninstall -y", "mpi4py")
        # Conda.pip("install --no-cache-dir", "mpi4py")

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
    end
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn "unable to build"
    else
        throw(e)
    end
end
