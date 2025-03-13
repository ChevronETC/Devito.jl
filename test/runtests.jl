using Devito

for testscript in ("serialtests.jl", "gencodetests.jl", "csymbolicstests.jl")
    include(testscript)
end

run(`mpirun -n 2 julia --code-coverage mpitests_2ranks.jl`)

if Devito.has_devitopro()
    @info "running devito pro tests"
    include("devitoprotests.jl")
else
    @info "not running devito pro tests"
end
