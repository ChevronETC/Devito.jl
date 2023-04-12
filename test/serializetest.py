from collections import namedtuple

import numpy as np
import pytest

from devito import (Eq, Grid, Operator)
from devitopro import *  # noqa
from devito.tools import make_tempdir
from devitopro.types.enriched import Disk
nt = 10
grid = Grid(shape=(9, 9))
time_dim = grid.time_dim

dumpdir = make_tempdir('aaa')

usave = TimeFunction(name='usave', grid=grid, save=nt, layers=Disk,
                        compression='bitcomp', serialization=dumpdir)

# Forward, dump everything to disk inside `dumpdir`
eq = Eq(usave, time_dim)
op0 = Operator(eq)
op0(time_M=nt-1)


# Now backward -- will fetch the data from disk
# We gonna use a different TimeFunction, to emulate the use case in which
# the overarching application, for whatever reason, can't reuse the
# same `usave`
v = TimeFunction(name='v', grid=grid)
vsave = TimeFunction(name='vsave', grid=grid, save=nt)

fetchdir = usave._fnbase

usave1 = TimeFunction(name='usave2', grid=grid, save=nt, layers=Disk,
                        compression='bitcomp', serialization=fetchdir)

eqns = [Eq(v, v.backward - 1),
        Eq(vsave, usave1)]
op1 = Operator(eqns)
op1(time_M=nt-1, time_m=0)