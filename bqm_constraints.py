from dimod import ConstrainedQuadraticModel, Binaries, ExactCQMSolver
import dimod
import numpy as np
import neal
import itertools
import pandas as pd
from copy import deepcopy
import json
import datetime
from dwave.preprocessing import FixVariablesComposite
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from benchmark_cqm  import standardCQM, solveBQM, convert_to_serializable


standardCQM=standardCQM()



from hybrid.reference.kerberos import KerberosSampler

from qiskit_optimization.algorithms import GurobiOptimizer
      
cqm = ConstrainedQuadraticModel()

keys=['x000', 'x001', 'x002', 'x003', 'x004', 'x010', 'x011', 'x012', 'x013', 'x014', 'x100', 'x101', 'x102', 'x103', 'x104', 'x110', 'x111', 'x112', 'x113', 'x114', 'y000', 'y001', 'y002', 'y003', 'y100', 'y101', 'y102', 'y103']

from dimod import Binary
bin_used = np.array(keys)

bin_used=np.vectorize(Binary)(bin_used)

bin_used=dict(enumerate(bin_used,1))
#bin_used.keys=keys

#cqm.set_objective(-70000*bin_used['x000']-14000*bin_used['x002']+(14000*bin_used['x002']*14000*bin_used['x001'])/2)

x000=Binary('x000')
x001=Binary('x001')
x002=Binary('x002')
x003=Binary('x003')
x004=Binary('x004')
x010=Binary('x010')
x011=Binary('x011')
x012=Binary('x012')
x013=Binary('x013')
x014=Binary('x014')
x100=Binary('x100')
x101=Binary('x101')
x102=Binary('x102')
x103=Binary('x103')
x104=Binary('x104')
x110=Binary('x110')
x111=Binary('x111')
x112=Binary('x112')
x113=Binary('x113')
x114=Binary('x114')
y000=Binary('y000')
y001=Binary('y001')
y002=Binary('y002')
y003=Binary('y003')
y100=Binary('y100')
y101=Binary('y101')
y102=Binary('y102')
y103=Binary('y103')

cqm.set_objective(- 700000*x001 - 1400000*x002 - 2100000*x003 - 2800000*x004 - 280000*x011
      - 560000*x012 - 840000*x013 - 1120000*x014 - 1600000*x101 - 3200000*x102
      - 4800000*x103 - 6400000*x104 - 640000*x111 - 1280000*x112 - 1920000*x113
      - 2560000*x114 + (1000000*x001*x001 + 4000000*x001*x002 + 6000000*x001*x003
      + 8000000*x001*x004 - 1000000*x001*x101 - 2000000*x001*x102
      - 3000000*x001*x103 - 4000000*x001*x104 + 4000000*x002*x002
      + 12000000*x002*x003 + 16000000*x002*x004 - 2000000*x002*x101
      - 4000000*x002*x102 - 6000000*x002*x103 - 8000000*x002*x104
      + 9000000*x003*x003 + 24000000*x003*x004 - 3000000*x003*x101
      - 6000000*x003*x102 - 9000000*x003*x103 - 12000000*x003*x104
      + 16000000*x004*x004 - 4000000*x004*x101 - 8000000*x004*x102
      - 12000000*x004*x103 - 16000000*x004*x104 + 160000*x011*x011
      + 640000*x011*x012 + 960000*x011*x013 + 1280000*x011*x014
      - 160000*x011*x111 - 320000*x011*x112 - 480000*x011*x113
      - 640000*x011*x114 + 640000*x012*x012 + 1920000*x012*x013 + 2560000*x012*x014
      - 320000*x012*x111 - 640000*x012*x112 - 960000*x012*x113
      - 1280000*x012*x114 + 1440000*x013*x013 + 3840000*x013*x014
      - 480000*x013*x111 - 960000*x013*x112 - 1440000*x013*x113
      - 1920000*x013*x114 + 2560000*x014*x014 - 640000*x014*x111
      - 1280000*x014*x112 - 1920000*x014*x113 - 2560000*x014*x114
      + 1000000*x101*x101 + 4000000*x101*x102 + 6000000*x101*x103
      + 8000000*x101*x104 + 4000000*x102*x102 + 12000000*x102*x103
      + 16000000*x102*x104 + 9000000*x103*x103 + 24000000*x103*x104
      + 16000000*x104*x104 + 160000*x111*x111 + 640000*x111*x112 + 960000*x111*x113
      + 1280000*x111*x114 + 640000*x112*x112 + 1920000*x112*x113
      + 2560000*x112*x114 + 1440000*x113*x113 + 3840000*x113*x114 + 2560000*x114*x114
      + 200*y000*y000 + 200*y000*y001 - 200*y000*y002 - 400*y000*y003
      - 400*y000*y100 - 200*y000*y101 + 200*y000*y102 + 400*y000*y103
      + 50*y001*y001- 100*y001*y002 - 200*y001*y003 - 200*y001*y100
      - 100*y001*y101 + 100*y001*y102 + 200*y001*y103 + 50*y002*y002
      + 200*y002*y003 + 200*y002*y100 + 100*y002*y101 - 100*y002*y102
      - 200*y002*y103 + 200*y003*y003 + 400*y003*y100 + 200*y003*y101
      - 200*y003*y102 - 400*y003*y103 + 200*y100*y100 + 200*y100*y101
      - 200*y100*y102 - 400*y100*y103 + 50*y101*y101 - 100*y101*y102
      - 200*y101*y103 + 50*y102*y102 + 200*y102*y103 + 200*y103*y103 )/2 + 6100000
      #+(x000*x001+x000*x002+x002*x003+x003*x004+x002*x001+x002*x004+x001*x003+x001*x004)*1e6 #bucket constraints
      +(x000*x001+x000*x002+x000*x003+x000*x004+x001*x002+x001*x003+x001*x004+x002*x003+x002*x004+x003*x004)*1e7
      +(x100*x101+x100*x102+x100*x103+x100*x104+x101*x102+x101*x103+x101*x104+x102*x103+x102*x104+x103*x104)*1e7
      +(x010*x011+x010*x012+x010*x013+x010*x014+x011*x012+x011*x013+x011*x014+x012*x013+x012*x014+x013*x014)*1e7
      +(x110*x111+x110*x112+x110*x113+x110*x114+x111*x112+x111*x113+x111*x114+x112*x113+x112*x114+x113*x114)*1e7
      +(y000*y001+y000*y002+y000*y003+y001*y002+y001*y003+y002*y003)*1e7
      +(y100*y101+y100*y102+y100*y103+y101*y102+y101*y103+y102*y103)*1e7

      )

#print("Unconstrained")
#solveBQM(cqm,False)

print("Only buckets")
#solution=solveBQM(cqm,False,2000,10,standardCQM)
#print("Feasible for standard CQM: {}".format(standardCQM.check_feasible(solution.sample)))


cqm.add_constraint(500*x001 + 1000*x002 + 1500*x003 + 2000*x004 == 500 ,'power_balance_plants_0')
cqm.add_constraint(200*x011 + 400*x012 + 600*x013 + 800*x014 == 200    ,'power_balance_plants_1')

cqm.add_constraint(500*x001 + 1000*x002 + 1500*x003 + 2000*x004      
                            + 200*x011 + 400*x012 + 600*x013 + 800*x014 == 700  ,'power_balance_all_times_0')                       
cqm.add_constraint(500*x101 + 1000*x102 + 1500*x103 + 2000*x104      
                            + 200*x111 + 400*x112 + 600*x113 + 800*x114 == 1600,'power_balance_all_times_1')


print("Power Balance")
#solveBQM(cqm,False,2000,10,standardCQM)

cqm.add_constraint(
10*y000 + 5*y001 - 5*y002 - 10*y003 - 10*y100 - 5*y101 + 5*y102
                 + 10*y103 >= -20,'transf_min_1_0')       

cqm.add_constraint(10*y000 + 5*y001 - 5*y002 - 10*y003 - 10*y100 - 5*y101 + 5*y102
                 + 10*y103 <= 20,'transf_max_1_0')

print("Transformers")
#solveBQM(cqm,True,2000,10,standardCQM)

cqm.add_constraint(- 100*x001 - 200*x002 - 300*x003 - 400*x004 - 40*x011
                       - 80*x012 - 120*x013 - 160*x014 - 100*x101 - 200*x102
                       - 300*x103 - 400*x104 - 40*x111 - 80*x112 - 120*x113
                       - 160*x114 + 2*y000 + 1*y001 - 1*y002 - 2*y003
                       + 2*y100 + 1*y101 - 1*y102 - 2*y103 <= -458,'branch_utilization_0')

cqm.add_constraint(- 100*x001 - 200*x002 - 300*x003 - 400*x004
                       - 20*x011 - 40*x012 - 60*x013 - 80*x014 - 100*x101
                       - 200*x102 - 300*x103 - 400*x104 - 20*x111 - 40*x112
                       - 60*x113 - 80*x114 + 4*y000 + 2*y001 - 2*y002
                       - 4*y003 + 4*y100 + 2*y101 - 2*y102 - 4*y103 <= 
                       -386,'branch_utilization_1')


print("Branch utilisation - full problem")
#solveBQM(cqm,False,2000,10,standardCQM)
solveBQM(cqm,True,2000,10,standardCQM,1e10)
#sampleset = dimod.ExactSolver().sample(bqm)
#print(sampleset)

#samplerHybrid = KerberosSampler().sample(bqm, max_iter=10, convergence=3)
#print("Hybrid sampler:")
#solution=samplerHybrid.first
#print(solution)
#print("Is feasible:")
#print(cqm.check_feasible(solution.sample))    

print("Gurobi solver")
optimizer = GurobiOptimizer(disp=False)
result = optimizer.solve(cqm)
print(result)

quit()
solutions={}

combinations = []
combinations.append([]) #Add the empty list, for having the original problem
for r in range(1, len(cqm.constraint_labels) + 1):
    combinations.extend(list(itertools.combinations(cqm.constraint_labels, r)))

exit()

# Print the combinations
for combo in combinations:
      cqm_simple=deepcopy(cqm)
      for constraint in combo:
            cqm_simple.remove_constraint(constraint)
      experimentLabel=", ".join(combo)
      bqm, invert = dimod.cqm_to_bqm(cqm)
      sampleset = sampler.sample(bqm, label='Example - CM - Scratch bqm '+experimentLabel)
      solution=sampleset.first
      feasible=cqm_simple.check_feasible(solution.sample)      
      solutions[experimentLabel]={"solution":solution.sample,"energy":solution.energy,"feasible":feasible,"fixed":False}
      sampleset = sampler.sample(bqm, label='Example - CM - Scratch bqm FIXED'+experimentLabel)
      sampler_fixed = FixVariablesComposite(sampler)
      solution=sampleset.first
      feasible=cqm_simple.check_feasible(solution.sample)
      solutions[experimentLabel+" fixed"]={"solution":solution.sample,"energy":solution.energy,"feasible":feasible,"fixed":True}

current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create the filename with the prefix
filename = f"{current_datetime}_data.json"

# Save the dictionary to a JSON file
with open(filename, "w") as json_file:
      json.dump(solutions, json_file, indent=4, default=convert_to_serializable)

      
          

samplesetExact = ExactCQMSolver().sample_cqm(cqm)

print(samplesetExact)