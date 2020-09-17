
# Devito.jl

| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

This is a Julia API for Devito.  it provides a Julia API's for a sub-set of Devito,
supporting `Grid`'s, `Function`'s, `TimeFunction`'s and `SparseTimeFunction`'s for both their
serial and domain decomposed MPI variants.

In addition it provides a Julia array interface for convenient and fast (copy-free) access to
the Devito numpy arrays.  For example,

```julia
using Devito

configuration!("language", "openmpi")
configuration!("mpi", false)

configuration() # show the Devito configuration

g = Grid(shape=(10,20,30))
f = Devito.Function(name="f", grid=g, space_order=8)
d = data(f)
d_with_halo = data_with_halo(f)
d_with_inhalo = data_with_inhalo(f)
```

In the above code listing:
* `d` is a view of the underlying numpy array that excludes Devito's exterior and interior halo padding
* `d_with_halo` is a view that includes Devito's exterior halo padding, but excludes its interior halo padding
* `d_with_inhalo` is a view that includes Divito's exterior and interior halo padding

If we are running with MPi turned on, then the `data`, `data_with_halo` and `data_with_inhalo` methods return
a view to an MPI domain distributed array.  This array can be gathered to the rank 0 MPI rank with the convert
method:
```julia
using Devito

configuration!("language", "openmpi")
configuration!("mpi", true)

g = Grid(shape=(10,20,30))
f = Devito.Function(name="f", grid=g, space_order=8)
d = data(f)
p = parent(d) # array local to this MPI rank
_d = convert(Array, d) # _d is an Array on MPI rank 0 gathered from `d` which is decomposed accross all MPI ranks
```

Please see the examples folder in this package for more details.

## Notes:
1. The Julia arrays returned by the `data`, `data_with_halo` and `data_with_inhalo` methods
are in the expected Julia/Fortran column major order (i.e. the first dimension is fast).
However, the tuples and arrays passed to the Devito Function, TimeFunction and SparseTimeFunction
methods are given in Python/C row major order.  This can cause some confusion at first.  Consider
the following example:

```julia
using Devito
g = Grid(shape=(10,11,12)) # 10 size of the slow dimension, and 12 is the size of the fast dimension.
f = Devito.Function(name="f", grid=g, space_order=8)
d = data(g) # size(d) is (12,11,10) where, as before, 10 is the size of the slow dimension, and 12 is the size of the fast dimension
```

If this caused too much confusion, we can, in the future, intercept the `shape` argument and reverse the direction of the tuple.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/Devito.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/Devito.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/Devito.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/Devito.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/Devito.jl/workflows/Tests/badge.svg
[build-status-url]: https://github.com/ChevronETC/Devito.jl/actions?query=workflow%3A"Tests"

[code-coverage-img]: https://codecov.io/gh/ChevronETC/Devito.jl/branch/master/graph/badge.svg
[code-coverage-results]: https://codecov.io/gh/ChevronETC/Devito.jl
