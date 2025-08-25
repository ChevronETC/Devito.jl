using MPI

run(`$(mpiexec()) -n 2 julia --code-coverage mpitests_2ranks.jl`)
run(`$(mpiexec()) -n 4 julia --code-coverage mpitests_4ranks.jl`)