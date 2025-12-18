# Ensure CondaPkg backend doesn't interfere
if !haskey(ENV, "JULIA_CONDAPKG_BACKEND")
    ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
end

using PythonCall

dpro_repo = get(ENV, "DEVITO_PRO", "")
which_devito = get(ENV,"DEVITO_BRANCH", "")

# Check if packages altready installed
# The assumption is that if the packages are already installed, the user 
# has already set up the environment and we don't need to do anything

# First thing first, is devito already installed
devito = try
    pyimport("devito")
    which_devito == ""
catch e
    @info "Devito not installed or broken"
    false
end

# Second, is devitopro installed
devitopro = try
    pyimport("devitopro")
    true
catch e
    @info "DevitoPRO not installed or broken"
    dpro_repo == ""
end

if (devito && devitopro)
    @info "Devito and DevitoPRO are already installed, no need to build"
    return
end

# Setup pip command. Automatically uses the Python seen by PythonCall.
function pip(pkg::String)
    sys = pyimport("sys")
    # sys.executable is a Py object; convert properly instead of String(sys.executable)
    exe = pyconvert(String, sys.executable)
    haspip = success(`$(exe) -m pip --version`)
    if !haspip
        @info "Bootstrapping pip (ensurepip)"
        try
            run(`$(exe) -m ensurepip --upgrade`)
        catch err
            @warn "ensurepip failed: $err"
        end
    end
    # Build command array safely (split returns substrings)
    cmd = String[exe, "-m", "pip", "install", "--no-cache-dir"]
    append!(cmd, split(pkg, ' '))
    run(Cmd(cmd))
end

# MPI4PY and optional nvidia requirements
function mpi4py(mpireqs)
    try
        ENV["CC"] = "nvc"
        ENV["CFLAGS"] = "-noswitcherror -tp=px"
        pip("-r $(mpireqs)requirements-mpi.txt")
        # If this succeeded, then we might need the extra nvidia python requirements
        pip("-r $(mpireqs)requirements-nvidia.txt")
    catch e
        # Default. Don't set any flag and use the default compiler
        delete!(ENV,"CFLAGS")
        ENV["CC"] = "gcc"
        pip("-r $(mpireqs)requirements-mpi.txt")
    end
    delete!(ENV,"CFLAGS")
    delete!(ENV,"CC")
end

# Install devito and devitopro
try
    ENV["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
    if dpro_repo != ""
        # Devitopro is available (Licensed). Install devitopro that comes with devito as a submodule. 
        # This way we install the devito version that is compatible with devitopro and we don't need to 
        # install devito separately.
        # Because devito is a submodule, pip fails to install it properly (pip does not clone with --recursive)
        # So we need to clone recursively, then install. And since julia somehow doesn't think submodules exists LibGit2 cannot clone
        # the submodules. So we need to clone it with git by hand.
        dir = abspath("$(tempname())-devitopro")
        Sys.which("git") === nothing && error("git is not installed")
        run(`git clone --recurse-submodules --depth 1 $(dpro_repo) $(dir)`)

        cd(dir)

        # Run install-devitopro.sh
        # @info("running install-devitopro.sh")
        # cmd_args = String["./install-devitopro.sh"]
        # run(Cmd(cmd_args))

        # Install devitopro
        @info("Install devitopro[extras]")
        pip("$(dir)[extras]")

        # Now all we need is mpi4py. It is straightforward to install except with the nvidia compiler that requires
        # extra flags to ignore some flags set by mpi4py
        mpi4py("$(dir)/submodules/devito/")
        rm(dir, recursive=true, force=true)

        # Make sure it imports
        pyimport("devitopro")
        pyimport("devito")
    else
        if which_devito != ""
            @info "Building devito from branch $(which_devito)"
            pip("devito[extras,tests]@git+https://github.com/devitocodes/devito@$(which_devito)")
            mpi4py("https://raw.githubusercontent.com/devitocodes/devito/$(which_devito)/")
        else
            @info "Building devito from latest release"
            pip("devito[extras,tests]")
            mpi4py("https://raw.githubusercontent.com/devitocodes/devito/main/")
        end
        # Make sure it imports
        pyimport("devito")
    end
    delete!(ENV, "PIP_BREAK_SYSTEM_PACKAGES")
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn "unable to build"
    else
        throw(e)
    end
end
