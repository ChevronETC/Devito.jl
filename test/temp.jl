using Devito, MPI, Test, PyCall

MPI.Init()

configuration!("mpi", true)

n = (11,10)

grid = Grid(shape=n, dtype=Float32)

b = Devito.Function(name="b", grid=grid, space_order=2)
b_data = data_with_halo(b)

_n = length(n) == 2 ? (15,14) : (16,15,14)

@test size(b_data) == _n

b_data_test = rand(Float32,_n)
copy!(b_data, b_data_test)

for rnk in 0:1
    if MPI.Comm_rank(MPI.COMM_WORLD) == rnk
        if rnk == 0
            if length(n) == 2
                @test parent(b_data) ≈ b_data_test[:,1:7]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,1:7]
                end
            end
            if rnk == 1
                if length(n) == 2
                    @test parent(b_data) ≈ b_data_test[:,8:14]
                else
                    @test parent(b_data) ≈ b_data_test[:,:,8:14]
                end
            end
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
end