
# Devito.jl

| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

Devito.jl is a Julia API supporting a sub-set of [devitoproject.org](https://www.devitoproject.org/).  It provides support for Devito `Grid`'s, `Function`'s, `TimeFunction`'s, `SparseTimeFunction`'s, `SubDomain`'s, `Dimension`'s, etc... for both their serial and domain decomposed MPI variants.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/Devito.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/Devito.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/Devito.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/Devito.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/Devito.jl/workflows/CI/badge.svg
[build-status-url]: https://github.com/ChevronETC/Devito.jl/actions?query=workflow%3ACI

[code-coverage-img]: https://codecov.io/gh/ChevronETC/Devito.jl/branch/master/graph/badge.svg
[code-coverage-results]: https://codecov.io/gh/ChevronETC/Devito.jl

## Installation

Devito.jl uses PythonCall.jl for Julia/Python interoperability.  In addition, it uses CondaPkg.jl for management of conda environments.  Please note that Devito has two versions: open-source and pro.  For the pro version please contact [DevitoCodes](https://www.devitocodes.com/) for more information.

### DevitoOpen
For the installation of open-source Devito, do:
```julia
using Pkg; Pkg.add(["Devito", "DevitoOpen"])
```
Please note that this will install a conda environment in the active Julia project.

### DevitoPro
For the installation of pro Devito on Linux, you will first need a copy of the python devito-pro repository, including its submodules (`git clone --recurse-submodules`), cloned to `/opt/devitopro`.  That folder will need both read and write permissions for the user that is installing DevitoPro, because devitopro is built inside it.  Once that is in place, do:
```julia
using Pkg; Pkg.add(["Devito", "DevitoPro"])
```
As with `DevitoOpen`, this will install a Conda environment in the active Julia project.  Please note that the same Julia project or conda environment cannot contain both `DevitoPro` and `DevitoOpen`.

If you subsequently update the contents of `/opt/devitopro`, then you will need to force a resolve for the project in a fresh Julia session:
```julia
using CondaPkg
CondaPkg.resolve(force=true) # same as: pkg> conda resolve --force
```
Then, restart Julia.  To confirm the new version, run:
```julia
using CondaPkg
```
and from the Julia package prompt run:
```
]conda run python -c "import importlib.metadata as m; print(m.version('devitopro'), m.version('devito'))"
```

## CondaPkg notes
It is important to note that the installation of python-devito does take some time (minutes), and devitopro takes longer since it is compiled.  Therefore, it may be useful to share conda environments between Julia projects so that each Julia project does not need its own installation.  For example, to have the current project use a shared environment named `@devito-open` (stored in `~/.julia/conda_environments/devito-open`), do:
```
pkg> add CondaPkg, PreferenceTools
julia> using CondaPkg, PreferenceTools
pkg> preference add CondaPkg env=@devito-open
```
This writes `env = "@devito-open"` under `[CondaPkg]` in the project's `LocalPreferences.toml`.  There is no separate install step for the shared environment: the first time CondaPkg resolves in a project with this setting (on `using Devito`, or explicitly with `CondaPkg.resolve()`), it creates `~/.julia/conda_environments/devito-open` and installs into it the Python dependencies declared by the project's packages (their `CondaPkg.toml` files).  Later projects with the same setting re-use it, and CondaPkg only installs what they add.  To build it right away, restart Julia so that the new preference is read, and then do:
```julia
using CondaPkg
CondaPkg.resolve()
CondaPkg.envdir()  # should be ~/.julia/conda_environments/devito-open
```
For `@devito-pro`, `/opt/devitopro` must be in place first.  A project's existing `.CondaPkg/env` is not moved into the shared environment; it is no longer used and can be deleted.  Some things to keep in mind:
* CondaPkg must be a direct dependency of the project (`pkg> add CondaPkg`).  Julia ignores preferences for packages that are only indirect dependencies, so without it the setting silently has no effect.
* Use one shared environment per backend (e.g. `@devito-open` and `@devito-pro`).  CondaPkg never removes packages from a shared environment, so letting DevitoOpen and DevitoPro projects share one would mix devito and devitopro.
* Don't set `env` in your global environment (`~/.julia/environments/v1.x`).  It would apply to every project that doesn't set its own, including projects that use the other backend.
* If a shared environment gets out of sync, run `CondaPkg.resolve(force=true)` from a project that uses it.

## Running the unit tests
The unit tests are in `DevitoOpen/test` and `DevitoPro/test`, and the `DevitoPro` tests also run the `DevitoOpen` suite.  CI follows the steps below; see `.github/workflows/ci.yml`, which also sets some Devito and Open MPI environment variables (for example, to allow more MPI ranks than cores).

1. Create a Julia project for testing, with your checkout of the packages and with `CondaPkg` and `MPIPreferences` as direct dependencies:
   ```
   pkg> activate ~/devito-tests
   pkg> dev path/to/Devito.jl/Devito path/to/Devito.jl/DevitoOpen
   pkg> add CondaPkg, MPIPreferences
   ```
   To test the pro backend, use `DevitoPro` instead of `DevitoOpen`.
2. Give the project a shared conda environment, for example `@devito-open` or `@devito-pro` (see [CondaPkg notes](#condapkg-notes)).  This is required: each `Pkg.test` run creates a new temporary project that uses the conda environment of the project you run it from, and step 3 needs that environment at a fixed path.
3. In a fresh Julia session, point MPI.jl at the conda environment's Open MPI.  This is the library that `mpi4py` uses, and since Julia and Python run in the same process, they must share it:
   ```julia
   using CondaPkg, MPIPreferences
   MPIPreferences.use_system_binary(;
       library_names=[CondaPkg.envdir("lib", "libmpi.so")],
       mpiexec=CondaPkg.envdir("bin", "mpiexec"),
   )
   ```
   This creates the shared environment if needed and writes the MPI paths to your project's `LocalPreferences.toml`, and `Pkg.test` passes them on to the tests.  Do not add them to the `LocalPreferences.toml` files in `DevitoOpen/test` or `DevitoPro/test`: the paths are machine-specific, and `Pkg.test` gives those files priority over your project.  The tests stop with an error if Julia and Python load different MPI libraries.  The same setting is needed in any project that uses Devito with MPI.jl.
4. Run the tests.  Conda's `mpicc` calls a conda compiler that is not installed, so point it at your system compilers.  Also use CondaPkg's offline mode: otherwise each `Pkg.test` run repeats the conda and pip installs, and although these normally download nothing, pip always rebuilds packages installed from a local folder, such as devitopro, which can take many minutes:
   ```julia
   using Pkg
   withenv("OMPI_CC" => "gcc", "OMPI_CXX" => "g++", "JULIA_CONDAPKG_OFFLINE" => "yes") do
       Pkg.test("DevitoOpen")  # or "DevitoPro"
   end
   ```
   Offline mode also skips any changes to the Python dependencies, so after changing a `CondaPkg.toml` or updating `/opt/devitopro`, run `CondaPkg.resolve(force=true)` once from your test project before running the tests again.

If `mpi4py` in your shared environment was installed by pip (older versions of Devito.jl built it against a system MPI), delete the environment (for example, `~/.julia/conda_environments/devito-open`) and let CondaPkg recreate it.