using Conda

Conda.add("pip")
pip = joinpath(Conda.BINDIR, "pip")
run(`$pip install cython`) 
run(`$pip install versioneer`) 
run(`$pip install git+https://github.com/devitocodes/devito.git`)
run(`$pip install mpi4py`)
run(`$pip install ipyparallel`)

#run(`$pip install devito`)
#run(`$pip install devito[extras]`)
