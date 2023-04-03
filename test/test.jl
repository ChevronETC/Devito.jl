using PyCall

@info "run"
run(`python test.py`)

@info "pyinclude"
@pyinclude("test.py")
