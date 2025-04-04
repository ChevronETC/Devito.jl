from pathlib import Path

from devito import *
from devito.tools import as_tuple

try:
    from devitopro import TimeFunction
    from devitopro.types.enriched import Disk, DiskHost
except ImportError:
    Disk = None


__all__ = ['serializedtimefunc', 'str2path', 'indexobj', 'ccode', 'subdom']


def serializedtimefunc(**kwargs):
    layers = kwargs.pop('layers', Disk)
    return TimeFunction(layers=layers, **kwargs)


def str2path(y):
    return Path(y)


def indexobj(x, *args):
    return x[args]


def ccode(x, filename):
    if filename == "":
        return print(x)
    else:
        with open(filename, 'w') as f:
            print(x,file=f)


class subdom(SubDomain):

    def __init__(self, name, instructions, *args, **kwargs):
        self.name = name
        self.instructions = as_tuple(instructions)
        super().__init__(*args, **kwargs)
    
    def define(self, dimensions):
        defines = {}
        for (d, i) in zip(dimensions, self.instructions):
            defines[d] = i
        
        return defines