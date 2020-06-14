using Conda

Conda.add("pip")
pip = joinpath(Conda.BINDIR, "pip")
run(`$pip install git+https://github.com/devitocodes/devito.git`)
run(`$pip install --upgrade sympy==1.5`)