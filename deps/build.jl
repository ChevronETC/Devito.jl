using PyCall

# dont use Conda!
# * assume python and pip are on path 
#       PyCall check below ensures python version >= 3.8
# * assume we have the Nvidia HPC_SDK version 22.3 

try
    # get python version number and assert > 3.8
    py"""import sys
    def v():
        return sys.version"""

    s = py"v()"
    v = parse.(Int,split(split(s," ")[1],"."))
    @assert v[1] >= 3
    @assert v[2] >= 8

    run(`rm -rf ./devito`)
    run(`git clone https://github.com/devitocodes/devito.git`)
    run(`pip install -r devito/requirements.txt`)
    run(`pip install -r devito/requirements-optional.txt`)

    # HPCX
    run(`sudo rm -f /opt/nvidia/hpc_sdk/Linux_x86_64/2022/comm_libs/mpi`)
    run(`sudo ln -sf /opt/nvidia/hpc_sdk/Linux_x86_64/2022/comm_libs/hpcx/latest/ompi /opt/nvidia/hpc_sdk/Linux_x86_64/2022/comm_libs/mpi`)

    # run(`pip uninstall -y mpi4py`)
    # run(`pip uninstall -y ipyparallel`)
    # run(`env CFLAGS="-noswitcherror" MPICC=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/comm_libs/mpi/bin/mpicc CC=nvc CFLAGS="-noswitcherror" $pip install --verbose --no-cache-dir mpi4py`)
    
catch e
    if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
        @warn unable to build
    else
        throw(e)
    end
end