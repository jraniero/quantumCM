# Global imports
import time
import numpy as np
import pandas as pd
import os
import dimod
import matplotlib.pyplot as plt
import neal

# Local imports
import sys


from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.algorithms import GurobiOptimizer
#from qiskit.optimization import QuadraticProgram
#from dwave.plugins.qiskit import DWaveMinimumEigensolver

from hybrid.samplers import SimulatedAnnealingSubproblemSampler
from hybrid.core import State
from dwave.system import LeapHybridSampler
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridBQMSampler
from dwave.preprocessing import FixVariablesComposite

milp_binary=dimod.lp.load("MILP_Model.lp")
milp_binary.change_vartype(vartype=dimod.SPIN,inplace=True)

sampler=neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(milp_binary, label='Example - CM')
sampler_fixed = FixVariablesComposite(sampler)
sampleset = sampler_fixed.sample(bqm_binary, fixed_variables={bqm_binary.variables[0]: 1}, num_reads=1000)