using PyCall

dpro_repo = get(ENV, "DEVITO_PRO", "")
# Example 1 using the mirror: DEVITO_PRO="https://chevron@dev.azure.com/chevron/ETC-ESD-COFIICloud/_git/devitopro-chevron"
# Example 2 using the original github: DEVITO_PRO="https://<username>:<token>@github.com/devitocodespro/devitopro-chevron"

dpro_tag = get(ENV, "DEVITO_PRO_TAG", "")
dpro_commit_sha = get(ENV, "DEVITO_PRO_COMMIT_SHA", "")
which_devito = get(ENV,"DEVITO_BRANCH", "")
force_install = get(ENV, "DEVITO_FORCE_INSTALL", false) == "1"

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

if (devito && devitopro) && !force_install
    @info "Devito and DevitoPRO are already installed, no need to build"
    exit(0)
end

# Setup pip command. This will automatically pickup whichever pip PyCall is setup with.
function pip(pkg::String)
    cmd_args = Vector{String}([PyCall.python, "-m", "pip", "install", "--no-cache-dir", split(pkg, " ")...])
    run(Cmd(cmd_args))
end

# Validate that a tag or branch actually exists on the remote before cloning.
# `git ls-remote` prints one line per matching ref and nothing when there is no
# match (while still exiting 0), so we check for empty output rather than the
# exit code.
function validate_remote_ref(repo, ref)
    out = read(`git ls-remote $(repo) $(ref)`, String)
    if isempty(strip(out))
        error("The tag/branch '$(ref)' was not found in the repository $(repo).")
    end
end

# MPI4PY and optional nvidia requirements
function mpi4py(mpireqs)
    try
        ENV["CC"] = "nvc"
        ENV["CFLAGS"] = "-noswitcherror -tp=px"
        pip("-r $(mpireqs)requirements-mpi.txt")
        # If this succeeded, the we might need the extra nvidia python requirements
        pip("-r $(mpireqs)requirements-nvidia.txt")
    catch e
        # Default. Don't set any flag an use the default compiler
        delete!(ENV,"CFLAGS")
        ENV["CC"] = "gcc"
        pip("-r $(mpireqs)requirements-mpi.txt")
    end
    delete!(ENV,"CFLAGS")
    delete!(ENV,"CC")
end

# Install devito and devitopro
try
    # Some python version don't like without --user so bypass it
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
        if dpro_tag != "" && dpro_commit_sha != ""
            @info "Both DEVITO_PRO_TAG ('$(dpro_tag)') and DEVITO_PRO_COMMIT_SHA ('$(dpro_commit_sha)') are set; the commit SHA will override the tag."
        end
        if dpro_tag == "" && dpro_commit_sha == ""
            run(`git clone --recurse-submodules --depth 1 $(dpro_repo) $(dir)`)
        elseif dpro_commit_sha != ""
            run(`git clone --recurse-submodules $(dpro_repo) $(dir)`)
            cd(dir)
            # A plain commit SHA cannot be checked via ls-remote, so validate it
            # against the freshly cloned repository before checking it out.
            if !success(pipeline(`git cat-file -t $(dpro_commit_sha)`, stdout=devnull, stderr=devnull))
                cd("..")
                rm(dir, recursive=true, force=true)
                error("The commit SHA '$(dpro_commit_sha)' was not found in the repository $(dpro_repo).")
            end
            run(`git checkout $(dpro_commit_sha)`)
            cd("..")
        elseif dpro_tag != ""
            # Fail early with a clear message if the tag/branch does not exist.
            validate_remote_ref(dpro_repo, dpro_tag)
            run(`git clone --recurse-submodules --branch $(dpro_tag) --depth 1 $(dpro_repo) $(dir)`)
        end

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
