
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

## CondaPkg notes
It is important to note that the installation of python-devito does take some time (minutes), and devitopro takes longer since it is compiled.  Therefore, it may be useful to share conda environments between Julia projects so that each Julia project does not need its own installation.  For example, to have the current project use a shared environment named `@devito-open` (stored in `~/.julia/conda_environments/devito-open`), do:
```
pkg> add CondaPkg, PreferenceTools
julia> using CondaPkg PreferenceTools
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

Finally, it is important to note that even with a shared environment, a fresh Julia project will always re-run the conda and pip installs.  These normally download nothing, but pip always rebuilds and reinstalls packages installed from a local folder, which is how devitopro is installed.  For DevitoPro this rebuild can take many minutes.  This is, in particular, noticeable when running unit tests since each run of `Pkg.test` creates a new temporary Julia project (which uses the `env` preference of the project you run `Pkg.test` from).  In this case we can avoid the costly rebuild with CondaPkg's offline mode:
```julia
using Pkg
withenv("JULIA_CONDAPKG_OFFLINE" => "yes") do
    Pkg.test("DevitoPro")
end
```
Offline mode only works with a shared environment; without one, the test project has no conda environment at all.  It also skips any changes to the Python dependencies, so after changing a `CondaPkg.toml` or updating `/opt/devitopro`, run `CondaPkg.resolve(force=true)` once from your project before testing offline again.