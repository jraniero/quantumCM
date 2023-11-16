#Reformulation of penalties as per IEEE paper of Giusseppe Colucci:
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10247202

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
from dimod import Binary



from hybrid.reference.kerberos import KerberosSampler

from qiskit_optimization.algorithms import GurobiOptimizer

def convert_to_serializable(obj):
    if isinstance(obj, np.int8):
        return int(obj)  # Convert int8 to int
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def solveBQM(cqm,cqmOrig,inspect=False,num_reads=5000,
annealing_time=40,
lagrange_multiplier=10):
      bqm, invert = dimod.cqm_to_bqm(cqm,lagrange_multiplier=lagrange_multiplier)
      bqmOrig,invertOrig  = dimod.cqm_to_bqm(cqmOrig,lagrange_multiplier=lagrange_multiplier)
      #print(cqm)
      qpu = DWaveSampler()
      samplesetQPU = EmbeddingComposite(qpu).sample(bqm,
                                          return_embedding=True,
                                          answer_mode="raw",
                                          num_reads=num_reads,
                                          annealing_time=annealing_time,
                                          auto_scale=True #Failing if set to false
                                          ,chain_strength=1e30
                                          ) 

                                          

      print("QPU annealer:")
      solution=samplesetQPU.first
      print(solution)
      print("Is feasible:")
      print(cqm.check_feasible(solution.sample))                          
      print("Is feasible Orig:")
      print(cqmOrig.check_feasible(solution.sample))            
      print(bqmOrig.energy(solution.sample))
      if inspect:
           dwave.inspector.show(samplesetQPU)

      sampler=neal.SimulatedAnnealingSampler()
      sampleset = sampler.sample(bqm, label='Example - CM - Scratch bqm',num_reads=10000)

      solution=sampleset.first

      print("Simulated annealer:")
      print(solution)
      print("Is feasible:")
      print(cqm.check_feasible(solution.sample))
      print("Is feasible Orig:")
      print(cqmOrig.check_feasible(solution.sample))
      print(bqmOrig.energy(solution.sample))
      
#bin_used.keys=keys

#cqm.set_objective(-70000*bin_used['x000']-14000*bin_used['x002']+(14000*bin_used['x002']*14000*bin_used['x001'])/2)

def originalCQM():
      cqm = ConstrainedQuadraticModel()
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
            - 2560000*x114 + (1000000*x001*x001 - 1000000*x001*x101 - 2000000*x001*x102
            - 3000000*x001*x103 - 4000000*x001*x104 + 4000000*x002*x002
            - 2000000*x002*x101
            - 4000000*x002*x102 - 6000000*x002*x103 - 8000000*x002*x104
            + 9000000*x003*x003 - 3000000*x003*x101
            - 6000000*x003*x102 - 9000000*x003*x103 - 12000000*x003*x104
            + 16000000*x004*x004 - 4000000*x004*x101 - 8000000*x004*x102
            - 12000000*x004*x103 - 16000000*x004*x104 + 160000*x011*x011
      
            - 160000*x011*x111 - 320000*x011*x112 - 480000*x011*x113
            - 640000*x011*x114 + 640000*x012*x012 
            - 320000*x012*x111 - 640000*x012*x112 - 960000*x012*x113
            + 1440000*x013*x013 
            - 480000*x013*x111 - 960000*x013*x112 - 1440000*x013*x113
            - 1920000*x013*x114 + 2560000*x014*x014 - 640000*x014*x111
            - 1280000*x014*x112 - 1920000*x014*x113 - 2560000*x014*x114
            + 1000000*x101*x101 + 4000000*x102*x102 + 9000000*x103*x103
            + 16000000*x104*x104 + 160000*x111*x111 + 640000*x112*x112 + 1440000*x113*x113 
            + 2560000*x114*x114
            + 200*y000*y000
            - 400*y000*y100 - 200*y000*y101 + 200*y000*y102 + 400*y000*y103
            + 50*y001*y001 - 200*y001*y100
            - 100*y001*y101 + 50*y002*y002
            + 200*y002*y003 + 200*y002*y100 + 100*y002*y101 - 100*y002*y102
            - 200*y002*y103 + 200*y003*y003 + 400*y003*y100 + 200*y003*y101
            - 200*y003*y102 - 400*y003*y103 + 200*y100*y100 + 200*y100*y101
            - 200*y100*y102 - 400*y100*y103 + 50*y101*y101 - 100*y101*y102
            - 200*y101*y103 + 50*y102*y102 + 200*y102*y103 + 200*y103*y103 )/2 + 6100000)

      cqm.add_constraint(x000 + x001 + x002 + x003 + x004 == 1,'bucket_p_0')
      cqm.add_constraint(x010 + x011 + x012 + x013 + x014 == 1,'bucket_p_1')
      cqm.add_constraint(x100 + x101 + x102 + x103 + x104 == 1,'bucket_p_2')
      cqm.add_constraint(x110 + x111 + x112 + x113 + x114 == 1,'bucket_p_3')
      cqm.add_constraint(y000 + y001 + y002 + y003 == 1       ,'bucket_s_0')
      cqm.add_constraint(y100 + y101 + y102 + y103 == 1       ,'bucket_s_1')

      return cqm

def penaltyReformQUBO():
      cqm = ConstrainedQuadraticModel()
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
            - 2560000*x114 + (1000000*x001*x001 - 1000000*x001*x101 - 2000000*x001*x102
            - 3000000*x001*x103 - 4000000*x001*x104 + 4000000*x002*x002
            - 2000000*x002*x101
            - 4000000*x002*x102 - 6000000*x002*x103 - 8000000*x002*x104
            + 9000000*x003*x003 - 3000000*x003*x101
            - 6000000*x003*x102 - 9000000*x003*x103 - 12000000*x003*x104
            + 16000000*x004*x004 - 4000000*x004*x101 - 8000000*x004*x102
            - 12000000*x004*x103 - 16000000*x004*x104 + 160000*x011*x011
      
            - 160000*x011*x111 - 320000*x011*x112 - 480000*x011*x113
            - 640000*x011*x114 + 640000*x012*x012 
            - 320000*x012*x111 - 640000*x012*x112 - 960000*x012*x113
            + 1440000*x013*x013 
            - 480000*x013*x111 - 960000*x013*x112 - 1440000*x013*x113
            - 1920000*x013*x114 + 2560000*x014*x014 - 640000*x014*x111
            - 1280000*x014*x112 - 1920000*x014*x113 - 2560000*x014*x114
            + 1000000*x101*x101 + 4000000*x102*x102 + 9000000*x103*x103
            + 16000000*x104*x104 + 160000*x111*x111 + 640000*x112*x112 + 1440000*x113*x113 
            + 2560000*x114*x114
            + 200*y000*y000
            - 400*y000*y100 - 200*y000*y101 + 200*y000*y102 + 400*y000*y103
            + 50*y001*y001 - 200*y001*y100
            - 100*y001*y101 + 50*y002*y002
            + 200*y002*y003 + 200*y002*y100 + 100*y002*y101 - 100*y002*y102
            - 200*y002*y103 + 200*y003*y003 + 400*y003*y100 + 200*y003*y101
            - 200*y003*y102 - 400*y003*y103 + 200*y100*y100 + 200*y100*y101
            - 200*y100*y102 - 400*y100*y103 + 50*y101*y101 - 100*y101*y102
            - 200*y101*y103 + 50*y102*y102 + 200*y102*y103 + 200*y103*y103 )/2 + 6100000
            
            # Penalties are added as quadratic terms in the cost function
            + 16000000*10*(x000 + x001 + x002 + x003 + x004 - 1)*(x000 + x001 + x002 + x003 + x004 - 1) #Bucket_p_0
            + 16000000*10*(x010 + x011 + x012 + x013 + x014 - 1)*(x010 + x011 + x012 + x013 + x014 - 1) #Bucket_p_0
            + 16000000*10*(x100 + x101 + x102 + x103 + x104 - 1)*(x100 + x101 + x102 + x103 + x104 - 1) #Bucket_p_2
            + 16000000*10*(x110 + x111 + x112 + x113 + x114 - 1)*(x110 + x111 + x112 + x113 + x114 - 1) #bucket_p_3
            + 16000000*10*(y000 + y001 + y002 + y003 - 1)*(y000 + y001 + y002 + y003 - 1) #bucket_s_0
            + 16000000*10*(y100 + y101 + y102 + y103 - 1)*(y100 + y101 + y102 + y103 - 1) #bucket_s_1
            )

      #cqm.add_constraint(x000 + x001 + x002 + x003 + x004 == 1,'bucket_p_0')
      #cqm.add_constraint(x010 + x011 + x012 + x013 + x014 == 1,'bucket_p_1')
      #cqm.add_constraint(x100 + x101 + x102 + x103 + x104 == 1,'bucket_p_2')
      #cqm.add_constraint(x110 + x111 + x112 + x113 + x114 == 1,'bucket_p_3')
      #cqm.add_constraint(y000 + y001 + y002 + y003 == 1       ,'bucket_s_0')
      #cqm.add_constraint(y100 + y101 + y102 + y103 == 1       ,'bucket_s_1')

     
      return cqm

print("Original CQM:")
originalcqm=originalCQM()
qubo=penaltyReformQUBO()
print(originalcqm)

print("Original CQM transformed into BQM:")
bqm, invert = dimod.cqm_to_bqm(originalcqm)
print(bqm)

print("Custom penalty QUBO:")
print(qubo)

solveBQM(qubo,originalcqm)

