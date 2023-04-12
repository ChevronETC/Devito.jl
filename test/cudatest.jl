using Devito

isdir("AAA") && rm("AAA", recursive=true, force=true)
dumpdir = mkdir("AAA")

compression = "bitcomp"

nt = 10
g = Grid(shape=(9, 9))
t = time_dim(g)

usave = TimeFunction(name="usave", grid=g, save=nt,
                        compression="bitcomp", serialization=dumpdir)

# Forward, dump everything to disk inside `dumpdir`
eq = Eq(usave, t)
op0 = Operator(eq)
apply(op0, time_M=nt-1)


# Now backward -- will fetch the data from disk
# We gonna use a different TimeFunction, to emulate the use case in which
# the overarching application, for whatever reason, can't reuse the
# same `usave`
v = TimeFunction(name="v", grid=g)
vsave = TimeFunction(name="vsave", grid=g, save=nt)

fetchdir = Devito.str2serial(Devito.serial2str(usave))

usave1 = TimeFunction(name="usave2", grid=g, save=nt,
                        compression="bitcomp", serialization=fetchdir)

eqns = [Eq(v, backward(v) - 1),
        Eq(vsave, usave1)]
op1 = Operator(eqns)
apply(op1, time_M=nt-1, time_m=0)