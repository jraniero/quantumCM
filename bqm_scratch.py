from dimod import ConstrainedQuadraticModel
import numpy as np
cqm = ConstrainedQuadraticModel()

keys=['x000', 'x001', 'x002', 'x003', 'x004', 'x010', 'x011', 'x012', 'x013', 'x014', 'x100', 'x101', 'x102', 'x103', 'x104', 'x110', 'x111', 'x112', 'x113', 'x114', 'y000', 'y001', 'y002', 'y003', 'y100', 'y101', 'y102', 'y103']

from dimod import Binary
bin_used = np.array(keys)

bin_used=np.vectorize(Binary)(bin_used)

bin_used=dict(enumerate(bin_used,1))
#bin_used.keys=keys

cqm.set_objective(-70000*bin_used['x000']-14000*bin_used['x002'])