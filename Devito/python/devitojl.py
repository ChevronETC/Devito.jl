# Kept in Python so Julia-free processes (e.g. devitopro decoupler workers) can unpickle these.
from devito import SparseFunction, SubDomain
from devito.tools import as_tuple


__all__ = ['subdom', 'coordslowsparse']


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


class coordslowsparse(SparseFunction):

        _sparse_position = 0
