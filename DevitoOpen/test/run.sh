julia ./csotests.jl
julia ./csymbolicstests.jl
julia ./devitoprotests.jl
julia ./gencodetests.jl
mpiexec -n 2 julia ./mpitests_2ranks.jl
julia ./runtests.jl
julia ./serialtests.jl
