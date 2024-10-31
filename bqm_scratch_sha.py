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




from hybrid.reference.kerberos import KerberosSampler

from qiskit_optimization.algorithms import GurobiOptimizer

def convert_to_serializable(obj):
    if isinstance(obj, np.int8):
        return int(obj)  # Convert int8 to int
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def solveBQM(cqm,inspect=False,num_reads=1000,annealing_time=40):
      bqm, invert = dimod.cqm_to_bqm(cqm)
      #print(cqm)
      qpu = DWaveSampler()
      samplesetQPU = EmbeddingComposite(qpu).sample(bqm,
                                          return_embedding=True,
                                          answer_mode="raw",
                                          num_reads=num_reads,
                                          annealing_time=annealing_time,
                                          auto_scale=True #Failing if set to false
                                          #,chain_strength=1e30
                                          ) 

                                          

      print("QPU annealer:")
      solution=samplesetQPU.first
      print(solution)
      print("Is feasible:")
      print(cqm.check_feasible(solution.sample))                          
      if inspect:
            dwave.inspector.show(samplesetQPU)

      sampler=neal.SimulatedAnnealingSampler()
      sampleset = sampler.sample(bqm, label='Example - CM - Scratch bqm',num_reads=10000)

      solution=sampleset.first

      print("Simulated annealer:")
      print(solution)
      print("Is feasible:")
      print(cqm.check_feasible(solution.sample))
      
cqm = ConstrainedQuadraticModel()

print("HEllo")

from dimod import Binary

for i in range(116756):
    globals()[f'x{i+1}'] =Binary(f'x{i+1}')


cqm.set_objective(x116756+\
2*x116755+\
3*x116754+\
4*x116753
)

print("Unconstrained")
#solveBQM(cqm,False)

cqm.add_constraint(x2553+x2548+x2552==0,'x2553')
cqm.add_constraint(x2554+x406+x2553==0,'x2554')
cqm.add_constraint(x2555+x2348+x2554==0,'x2555')
cqm.add_constraint(x2556+x2348+x2553==0,'x2556')

cqm.add_constraint(x2558+x2553+x2557==0,'x2558')
cqm.add_constraint(x2559+x405+x2558==0,'x2559')
cqm.add_constraint(x2560+x2347+x2559==0,'x2560')
cqm.add_constraint(x2561+x2347+x2558==0,'x2561')



print("Branch utilisation - full problem")
solveBQM(cqm,False,2000,10)
sampleset = dimod.ExactSolver().sample(bqm)
print(sampleset)

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