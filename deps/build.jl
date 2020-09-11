using Conda

Conda.add("pip")
pip = joinpath(Conda.BINDIR, "pip")
run(`$pip install git+https://github.com/devitocodes/devito.git`)
run(`$pip install mpi4py`)
run(`$pip install ipyparallel`)
run(`$pip install --upgrade sympy'<'1.6`)

#run(`$pip install devito`)
#run(`$pip install devito[extras]`)
