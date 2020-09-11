# Devito.jl

Package Devito is a Julia wrapper for the Devito stencil compiler. The procedure for constructing a stencil operator consists of five parts:

1. Construction of a Grid object containing grid size and spacing information on which the stencil will operate
2. Construction of TimeFunction or Function objects that hold the actual arrays on which the stencil will operate
3. Construction of SparseTimeFunction objects that hold source and receiver data for injection and retrieval during the stencil operation
4. Construction of Eqn objects (equation objects) that specify the equations that the stencil operator will carry out
5. Construction of the operator object which generates the low level C code fo rthe stencil operator

Following this the operator may be executed. An example of these five steps is detailed below

1. Definition of the Grid object

The `Grid` object is specified by initializing extent, spacing, and origin tuples in the constructor. Dimension objects contain the 
abstract spacing variables used by SymPy in specifying abstract equations in the stencil definition

```julia
dt=1.
shpe=(20,20)
spacing=(5.,5.)
origin=(0.,0.)
extent=(shpe.-1).*spacing
spacet=Constant(name="h_t", dt) 
t=TimeDimension(name="t",spacing=spacet)
spacex=Constant(name="h_x", spacing[1]) 
x=SpaceDimension(name="x",spacing=spacex)
spacez=Constant(name="h_z", spacing[2])
z=SpaceDimension(name="z",spacing=spacez)
grid=Grid(extent=extent, shape=shpe, origin=origin, dimensions=(x,z), time_dimension=t)
```

Note that he dimensions are passed in column-major order, in alignment with the Devito Python implementation. For all other tuples involving dimensions in Julia, the ordering is row-major.

2. Construction of time and space functions

Parameters on the grid are specified using Function objects, while time dependent fields are specified using TimeFunction objects, 
as in this 2D elastic example:

```julia
so=4
bx=Devito.Function(name= "bx" ,grid=grid, staggered=x,     space_order=so)
bz=Devito.Function(name= "bz" ,grid=grid, staggered=z,     space_order=so)
c11=Devito.Function(name="c11",grid=grid, space_order=so)
c33=Devito.Function(name="c33",grid=grid, space_order=so)
c55=Devito.Function(name="c55",grid=grid, staggered=(x,z), space_order=so)
c13=Devito.Function(name="c13",grid=grid, space_order=so)
data(bx).=mp[:,:,1]
data(bz).=mp[:,:,2]
data(c11).=mp[:,:,3]
data(c33).=mu[:,:,2]
data(c55).=mp[:,:,4]
data(c13).=mp[:,:,5]
vx=TimeFunction(name="vx",      grid=grid,space_order=so, staggered=x, time_order=1)
vz=TimeFunction(name="vz",      grid=grid,space_order=so, staggered=z, time_order=1)
tauxx=TimeFunction(name="tauxx",grid=grid,space_order=so,                  time_order=1)
tauzz=TimeFunction(name="tauzz",grid=grid,space_order=so,                  time_order=1)
tauxz=TimeFunction(name="tauxz",grid=grid,space_order=so, staggered=(x,z), time_order=1)
```
  
In this example, the `data()` function returns a view to the Function's internal data, which is then initialized from an externally defined arrays mp and mu.

3. Construction of  SparseTimeFunctions

`SparseTimeFunctions` are used to inject source and retrieve receiver information during the stencil operations

```julia
src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=nt)
src_coords = coordinates(src)
src_coords .= [625.0; 5.0]
src_data = data(src)
src_data.= ricker(nt,dt,f0)
src_term = inject(src; field=forward(tauxx), expr=src)
```

In this example, the source is created with an external function `ricker()`, which is then used to initialize the SparseTimeFunction data that will be injected into the
time function `forward(tauxx)`

4. Construction of  stencil equations

Stencil equations are created using the Eqn constructor 

```julia
eqvx = Eq(forward(vx), vx + dt*bx*dx(tauxx) + dt*bx*dz(tauxz, x0=z) - dt*damp*vx)
eqvz = Eq(forward(vz), vz + dt*bz*dx(tauxz, x0=x) + dt*bz*dz(tauzz) - dt*damp*vz)
```

In this particular fragment from an elastic 2D example, damp is an externally defined array for damping at the boundaries, vx and vz are particle velocities, and the tau variables are the stresses 

5. Construction of the operator

Construction of the operator requires a list containing all objects created using the `inject()`, `interpolate()`, and `Eq()` functions:

```julia
op=Operator(op_list, subs=spacing_map(grid))
apply(op)
```

where op_list contains the objects comprising the stencil.

In general, the wrapping of Devito functionality uses the same function and argument names as in the original Python implementation, with 
python class members being accessed in Julia through functions having the same name as the member, and taking the class object as the first argument.

For more details, please refer to the Devito website https://github.com/devitocodes/devito


