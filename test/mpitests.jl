using MPI

withenv("DEVITO_AUTOPADDING" => "0") do
    run(`$(mpiexec()) -n 2 julia --code-coverage mpitests_2ranks.jl`)
    run(`$(mpiexec()) -n 4 julia --code-coverage mpitests_4ranks.jl`)
end