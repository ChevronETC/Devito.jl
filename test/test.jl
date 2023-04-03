using PyCall

@info "run"
run(`python test.py`)

@info "pyinclude"
@pyinclude("test.py")


using Devito
@info "using Devito"
@info "run"
run(`python test.py`)

@info "pyinclude"
@pyinclude("test.py")