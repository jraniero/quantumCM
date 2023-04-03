# Global imports
import time
import numpy as np
import pandas as pd
import os
import dimod

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

P = 2 # power plants
TRAN = 1 # transformers
T = 2 # timesteps
BRANCH = 2 # branches
I = 5 # power plants buckets
J = 4 # transformers buckets
CP_COEF = 1 # power plant coefficient
CS_COEF = 1 # transformer coefficient
CDIFF_COEF = 1 # energy coefficient
V_P = [[0, 500, 1000, 1500, 2000], [0, 200, 400, 600, 800]] # power plant buckets
V_S = [[-10, -5, +5, +10]] # transformer buckets
# power balance inputs
P_0_T = [700, 1600] # original power output (all plants) // p_0[t]
P_0_P_T = [[500, 1000], [200, 600]] # original power output (per plant) first time // p[p][t]
# producer change inputs
P_MIN = -1000 # delta power min
P_MAX = 1000 # delta power max
# transformer change inputs
S_MIN = -20 # delta transformer min
S_MAX = +20 # delta transformer max
# branch utilization inputs
S_0_TRAN_T = [[+10, +10]] # original trans config // s[tran][t] // this should be [-32, 32]
I_0 = [10, 15] # initial 
I_B = [40, 35]
S_P_B = [[1, 2], [1, 1]]
S_TRAN_B = [[1, 4], [2, 3]]
# time on/off inputs
T_ON = 1
T_OFF = 1

# other inputs variables for the algorithm
SOLVER = "SimulatedAnnealingSolver"
TUNING = False


def _create_mip_objective(p, t, tran, i, j, cp_coef, cs_coef, cd_coef, v_p, v_s, p_0_t):
    """
    Create the MIP statement for the congestion management problem based on the
    objective function of power redispatch, transformer change and deviation from
    original schedule.

    Parameters
    ----------
    p : int
        Set of power plants.
    t : int
        Number of timesteps.
    tran : int
        Set of transformers.
    i : int
        Set of n power buckets.
    j : int
        Set of m transformer buckets.
    cp_coef : float
        Power redispatch penalty.
    cs_coef : float
        Transformer change penalty.
    cd_coef : float
        Power schedule deviation penalty.
    v_p : list (int)
        Value of corresponding power bucket.
    v_s : list (int)
        Value of corresponding transformer bucket.
    p_0_t : list (int)
        Original power output (sum across all plants) [MW].

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management.
    lst_vars : dict
        Dictionary with the binary variables of the problem.

    """

    mip = Model()

    lst_vars = dict()

    for a in range(t):
        for b in range(p):
            for c in range(i):
                lst_vars[f'x{a}{b}{c}'] = mip.binary_var(f'x{a}{b}{c}')
    for a in range(t):
        for b in range(tran):
            for c in range(j):
                lst_vars[f'y{a}{b}{c}'] = mip.binary_var(f'y{a}{b}{c}')

    power_p = list()
    power_s = list()

    for a in range(t):
        plant_bucket = list()
        for b in range(p):
            bucket = list()
            for c in range(i):
                bucket.append(v_p[b][c]*lst_vars[f'x{a}{b}{c}'])
            plant_bucket.append(sum(bucket))
        power_p.append(plant_bucket)

    for a in range(t):
        plant_bucket = list()
        for b in range(tran):
            bucket = list()
            for c in range(j):
                bucket.append(v_s[b][c]*lst_vars[f'y{a}{b}{c}'])
            plant_bucket.append(sum(bucket))
        power_s.append(plant_bucket)

    delta_p = list()
    delta_s = list()
    delta_p_diff = list()

    for a in range(len(power_p)-1,0,-1):
        aux = list()
        for b in range(p):
            aux.append(power_p[a][b]-power_p[a-1][b])
        delta_p.append(aux)

    for a in range(len(power_s)-1,0,-1):
        aux = list()
        for b in range(tran):
            aux.append(power_s[a][b]-power_s[a-1][b])
        delta_s.append(aux)

    for a in range(len(power_p)):
        aux = list()
        for b in range(p):
            aux.append(p_0_t[a]-power_p[a][b])
        delta_p_diff.append(aux)


    cp = list()
    cs = list()
    cdiff = list()

    for a in delta_p:
        aux = list()
        for b in range(p):
            aux.append(a[b]**2)
        cp.append(cp_coef*sum(aux))

    for a in delta_s:
        aux = list()
        for b in range(tran):
            aux.append(a[b]**2)
        cs.append(cs_coef*sum(aux))

    for a in delta_p_diff:
        aux = list()
        for b in range(p):
            aux.append(a[b]**2)
        cdiff.append(cd_coef*sum(aux))

    final_obj = sum(cp) + sum(cs) + sum(cdiff)

    mip.minimize(final_obj)

    mip = from_docplex_mp(mip)

    return mip, lst_vars

def _mip_const_buckets(mip, t, p, tran, i, j):
    """
    Constraint to fulffill that just one value is selected for each bucket.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem.
    t : int
        Number of timesteps.
    p : int
        Number of power plants.
    tran : int
        Number of transformers.
    i : int
        Number of power buckets.
    j : int
        Number of power buckets.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (bucket constraint added).

    """

    const_p = list()
    const_s = list()

    for a in range(t):
        for b in range(p):
            aux = dict()
            for c in range(i):
                aux[f'x{a}{b}{c}'] = 1
            const_p.append(aux)

    for a in range(t):
        for b in range(tran):
            aux = dict()
            for c in range(j):
                aux[f'y{a}{b}{c}'] = 1
            const_s.append(aux)
    
    
    for il,a in enumerate(const_p):
        mip.linear_constraint(linear=a, sense='==', rhs=1,
                                name=f'bucket_p_{il}')

    for il,a in enumerate(const_s):
        mip.linear_constraint(linear=a, sense='==', rhs=1,
                                name=f'bucket_s_{il}')

    return mip

def _mip_const_power_balance(mip, p_0_t, p_0_p_t, t, p, i, v_p):
    """
    Two constraints to fulfill:
        1. Starting power in each plant is preserved from the day before for
        every power plant.
        2. All plants preserve the power in each timestep.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the model.
    p_0_t : list (int)
        Original power output for all power plants.
    p_0_p_t : list (int)
        Original power output per plant.
    t : int
        Number of timesteps.
    p : int
        Number of power plants.
    i : int
        Number of n power buckets.
    v_p : list (float)
        Value of corresponding power bucket.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (power balance constraint added).

    """

    pow_bal_1 = list()
    pow_bal_2 = list()

    for a in range(t):
        aux_2 = dict()
        for b in range(p):
            aux_1 = dict()
            for c in range(i):
                aux_1[f'x{0}{b}{c}'] = v_p[b][c]
                aux_2[f'x{a}{b}{c}'] = v_p[b][c]
            if a == 0:
                pow_bal_1.append(aux_1)
        pow_bal_2.append(aux_2)

    for a in range(len(pow_bal_1)):
        mip.linear_constraint(linear=pow_bal_1[a], sense='==', rhs=p_0_p_t[a][0],
                               name=f'power_balance_plants_{a}')
    
    for a in range(len(pow_bal_2)):
        mip.linear_constraint(linear=pow_bal_2[a], sense='==', rhs=p_0_t[a],
                               name=f'power_balance_all_times_{a}')

    return mip

def _mip_const_transformer(mip, s_min, s_max, v_s, tran, t, j):
    """
    Constraint to fulfill the minimum and maximum values the transformer changes
    can have at every timestep.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem.
    s_min : int
        Minimum possible value of the transformer change.
    s_max : int
        Maximum possible value of the transformer change.
    v_s : list (float)
        Value of corresponding transformer bucket.
    tran : int
        Set of number of transformers.
    t : int
        Number of timesteps.
    j : int
        Number of transformer buckets.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (transformer change constraint added).

    """

    const_min = list()
    const_max = list()

    for a in range(t-1,0,-1):
        for b in range(tran):
            aux_min = dict()
            aux_max = dict()
            for c in range(j):
                aux_min[f'y{a}{b}{c}'] = v_s[b][c]
                aux_min[f'y{a-1}{b}{c}'] = -v_s[b][c]
                aux_max[f'y{a}{b}{c}'] = v_s[b][c]
                aux_max[f'y{a-1}{b}{c}'] = -v_s[b][c]
            const_min.append(aux_min)
            const_max.append(aux_max)

    for a in const_min:
        mip.linear_constraint(linear=a, sense='>=', rhs=s_min,
                               name=f'transf_min_{list(a.keys())[0][1]}_{list(a.keys())[0][2]}')

    for a in const_max:
        mip.linear_constraint(linear=a, sense='<=', rhs=s_max,
                               name=f'transf_max_{list(a.keys())[0][1]}_{list(a.keys())[0][2]}')

    return mip

def _mip_const_branch(mip, branch, i_0, s_p_b, s_tran_b, p_0_p_t, s_0_tran_t,
                       i_b, v_p, v_s, t, p, tran, i, j):
    """
    Constraint to fulfill what branches are being used in the congestion
    management problem.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the use case.
    branch : int
        Set of branches.
    i_0 : int
        Initial current in the branch.
    s_p_b : list (float)
        Sensitivity value for power change on the branch.
    s_tran_b : list (float)
        Sensitivity value for transformer change on the branch.
    p_0_p_t : list (int)
        Original power output per plant.
    s_0_tran_t : list (int)
        Original trnasformer configuration.
    i_b : list (int)
        Max current on hte branch.
    v_p : list (int)
        Value of corresponding power bucket.
    v_s : list (int)
        Value of corresponding transformer bucket.
    t : int
        Number of timesteps.
    p : int
        Number of power plants.
    tran : int
        Set of trnasformers.
    i : int
        Number of n power buckets.
    j : int
        Number of m power buckets.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (branch utilization constraint added).

    """

    const_p = list()
    const_s = list()
    constant_p = list()
    constant_s = list()
    const_final = list()
    constant_final = list()

    for br in range(branch):
        aux_list_p = list()
        aux_p = dict()
        for a in range(t):
            for b in range(p):
                aux_list_p.append(s_p_b[b][br]*p_0_p_t[b][a])
                for c in range(i):
                    aux_p[f'x{a}{b}{c}'] = -s_p_b[b][br]*v_p[b][c]
        constant_p.append(sum(aux_list_p))
        const_p.append(aux_p)

    for br in range(branch):
        aux_list_s = list()
        aux_s = dict()
        for a in range(t):
            for b in range(tran):
                aux_list_s.append(s_tran_b[b][br]*s_0_tran_t[b][a])
                for c in range(j):
                    aux_s[f'y{a}{b}{c}'] = -s_tran_b[b][br]*v_s[b][c]
        constant_s.append(sum(aux_list_s))
        const_s.append(aux_s)

    for br in range(branch):
        const_p[br].update(const_s[br])
        const_final.append(const_p[br])
        constant_final.append(constant_p[br]+constant_s[br])
        mip.linear_constraint(linear=const_final[br], sense='<=',
                               rhs=i_b[br]-i_0[br]-constant_final[br],
                               name=f'branch_utilization_{br}')

    return mip

def _mip_const_producers_change(mip, p_min, p_max, v_p, p, t, i):
    """
    Constraint to fulfill the power change for each plant and timestep.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion manangement problem.
    p_min : int
        Minimum power change allowed.
    p_max : int
        Maximum power changed allowed.
    v_p : list (int)
        Value of corresponding power buckets.
    p : int
        Number of power plants.
    t : int
        Number of timesteps.
    i : int
        Number of n power buckets.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (adding producer change constraint).

    """

    const_min = list()
    const_max = list()

    for a in range(t-1,0,-1):
        for b in range(p):
            aux_min = dict()
            aux_max = dict()
            for c in range(i):
                aux_min[f'x{a}{b}{c}'] = v_p[b][c]
                aux_min[f'x{a-1}{b}{c}'] = -v_p[b][c]
                aux_max[f'x{a}{b}{c}'] = v_p[b][c]
                aux_max[f'x{a-1}{b}{c}'] = -v_p[b][c]
            const_min.append(aux_min)
            const_max.append(aux_max)
    
    for il,a in enumerate(const_min):
        mip.linear_constraint(linear=a, sense='>=', rhs=p_min,
                               name=f'power_min_{il}')

    for il,a in enumerate(const_max):
        mip.linear_constraint(linear=a, sense='<=', rhs=p_max,
                               name=f'power_max_{il}')

    return mip


def _mip_const_producers_time(mip, t_off, t_on, t, p, i):
    """
    Constraint to fulfill the operative time on/of for each power plant and timestep.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem.
    t_off : int
        Minimum off times for a power plant.
    t_on : int
        Minimum on times for a power plant.
    t : int
        Number of timesteps.
    p : int
        Numer of power plants.
    i : int
        Set of n power buckets.

    Returns
    -------
    mip : docplex.mp.model.Model
        MIP formulation of the congestion management problem (adding producer time constraint).

    """

    const_off = list()
    const_on = list()

    for a in range(t-1,0,-1):
        for b in range(p):
            aux_off = dict()
            aux_off[f'x{a-1}{b}{0}'] = 1
            aux_off[f'x{a}{b}{0}'] = -t_off
            for u in range(2,t_off):
                if u <= a:
                    aux_off[f'{a-u}{b}{0}'] = -1
            const_off.append(aux_off)

    for a in range(t-1,0,-1):
        for b in range(p):
            aux_on = dict()
            aux_on[f'x{a}{b}{0}'] = 1
            aux_on[f'x{a-1}{b}{0}'] = -t_on
            for u in range(2,t_on):
                for c in range(1,i):
                    if u <= a:
                        aux_on[f'x{a-u}{b}{c}'] = -1
            const_on.append(aux_on)
    
    for il,it in enumerate(const_off):
        mip.linear_constraint(linear=it, sense='<=', rhs=1,
                                name=f'producer_off_{il}')

    for il,it in enumerate(const_on):
        mip.linear_constraint(linear=it, sense='<=', rhs=1,
                                name=f'producer_on_{il}')

    return mip

def _qubo_converter(mip, penalty):
    """
    Converting the MIP problem to a QUBO formulation. Introducing all constraints
    into the objective function; and converting it to an Ising hamiltonian.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        MIP problem of the congestion management problem.
    penalty : int
        Penalty associated to all constraints.

    Returns
    -------
    qubo : docplex.mp.model.Model
        QUBO formulation of the congestion management problem.
    num_vars : int
        Number of variables of the problem.
    hamiltonian : docplex.mp.model.Model
        Ising hamiltonian of the problem.
    offset : numpy.float
        Offset of the Ising hamiltonian. Indpendent term.

    """
    if penalty != None:
        convt = QuadraticProgramToQubo(penalty=penalty)
    else:
        convt = QuadraticProgramToQubo()

    qubo = convt.convert(mip)

    num_vars = qubo.get_num_vars()
    hamiltonian, offset = qubo.to_ising()

    return qubo, num_vars, hamiltonian, offset

def _gurobi_solver_(mip):
    """
    Gurobi solver to solve the mip formulation.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        mip formulation of the problem.

    Returns
    -------
    result : qiskit_optimization.algorithms.optimization_algorithm.OptimizationResult
        Result of the problem.

    """

    optimizer = GurobiOptimizer(disp=False)
    result = optimizer.solve(mip)

    print(result,'\n')
    print(result.variables_dict, '\n')

    return result

def _check_result_(lst_vars, mip, result):
    """
    Checking the result obtained from the solver.

    Parameters
    ----------
    mip : docplex.mp.model.Model
        mip formulation of the problem.
    result : qiskit_optimization.algorithms.optimization_algorithm.OptimizationResult
        Result of the solver.

    Returns
    -------
    feasible: Boolean
        Whether the obtained solution is feasible or not.

    """

    solution = list()
    for i, j in result.variables_dict.items():
        if i in lst_vars:
            solution.append(round(j))
    print(solution)
    feasible = mip.is_feasible(solution)

    print('\nIs the solution feasible?:', mip.is_feasible(solution), '\n')

    return feasible

def _print_solution(solution, bucket_p, bucket_s):
    """
    Printing the solution.

    """

    for i,j in solution.items():
        if j == 1:
            if i[0]=='x':
                print(f'Power plant {i[2]} in timestep {i[1]}: {bucket_p[int(i[2])][int(i[3])]} MW')
            elif i[0]=='y':
                print(i)
                print(f'Transformer {i[2]} in timestep {i[1]}: {bucket_s[int(i[2])][int(i[3])]}\n MW')

def _write_csv(P, TRAN, T, BRANCH, I, J, CP_COEF, CS_COEF, CDIFF_COEF, V_P, V_S,
              P_0_T, P_0_P_T, P_MIN, P_MAX, S_MIN, S_MAX, S_0_TRAN_T, I_0, I_B,
              S_P_B, S_TRAN_B, T_ON, T_OFF, backend, feasible, time, execution_time,
              eigenvalue, num_vars, parameters, solution, path='.'):
    """
    Writing function.

    """

    if os.path.exists(os.path.join(path, 'Tests_SA_Congestion_Management.csv')):
        df = pd.read_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'))
        ide = df['ide'].max()
        ide = ide + 1
        flag = False
    else:
        flag = True
        ide = 0

    df = pd.DataFrame(np.array([[ide, P, TRAN, T, BRANCH, I, J, CP_COEF, CS_COEF, CDIFF_COEF, V_P, V_S,
                                 P_0_T, P_0_P_T, P_MIN, P_MAX, S_MIN, S_MAX, S_0_TRAN_T, I_0, I_B,
                                 S_P_B, S_TRAN_B, T_ON, T_OFF, backend, feasible,
                                 time, execution_time, eigenvalue, num_vars, parameters, solution]]),
                      columns=['ide', "# power plants", "# transformers", "# timesteps",
                               "# branches", "# power buckets", "# transfomer buckets",
                               "Penalty cp", "Penalty cs", "Penalty cdiff", "Power buckets",
                               "Transformer buckets", "Original power output (all)",
                               "Original power output per plant", "Minimum power change",
                               "Maximum power change", "Minimum transformer change",
                               "Maximum transformer change", "Original transformer config",
                               "Initial current in the branch", "Maximum current in the branch",
                               "Power sensitivity", "Transformer sensitivity", "# times on", "# times off",
                               "Backend", "Feasibility", "Total time", "Execution time",
                               "Eigenvalue", "# variables", "Parameters", "Solution"])

    if flag:
        df.to_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'), header=True, mode='a', index=False)
    else:
        df.to_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'), header=False, mode='a', index=False)
    

def _print_solution(solution, bucket_p, bucket_s):
    """
    Printing the solution.

    """

    for i,j in solution.items():
        if j == 1:
            if i[0]=='x':
                print(f'Power plant {i[2]} in timestep {i[1]}: {bucket_p[int(i[2])][int(i[3])]} MW')
            elif i[0]=='y':
                print(i)
                print(f'Transformer {i[2]} in timestep {i[1]}: {bucket_s[int(i[2])][int(i[3])]}\n MW')

def _write_csv(P, TRAN, T, BRANCH, I, J, CP_COEF, CS_COEF, CDIFF_COEF, V_P, V_S,
              P_0_T, P_0_P_T, P_MIN, P_MAX, S_MIN, S_MAX, S_0_TRAN_T, I_0, I_B,
              S_P_B, S_TRAN_B, T_ON, T_OFF, backend, feasible, time, execution_time,
              eigenvalue, num_vars, parameters, solution, path='.'):
    """
    Writing function.

    """

    if os.path.exists(os.path.join(path, 'Tests_SA_Congestion_Management.csv')):
        df = pd.read_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'))
        ide = df['ide'].max()
        ide = ide + 1
        flag = False
    else:
        flag = True
        ide = 0

    df = pd.DataFrame(np.array([[ide, P, TRAN, T, BRANCH, I, J, CP_COEF, CS_COEF, CDIFF_COEF, V_P, V_S,
                                 P_0_T, P_0_P_T, P_MIN, P_MAX, S_MIN, S_MAX, S_0_TRAN_T, I_0, I_B,
                                 S_P_B, S_TRAN_B, T_ON, T_OFF, backend, feasible,
                                 time, execution_time, eigenvalue, num_vars, parameters, solution]]),
                      columns=['ide', "# power plants", "# transformers", "# timesteps",
                               "# branches", "# power buckets", "# transfomer buckets",
                               "Penalty cp", "Penalty cs", "Penalty cdiff", "Power buckets",
                               "Transformer buckets", "Original power output (all)",
                               "Original power output per plant", "Minimum power change",
                               "Maximum power change", "Minimum transformer change",
                               "Maximum transformer change", "Original transformer config",
                               "Initial current in the branch", "Maximum current in the branch",
                               "Power sensitivity", "Transformer sensitivity", "# times on", "# times off",
                               "Backend", "Feasibility", "Total time", "Execution time",
                               "Eigenvalue", "# variables", "Parameters", "Solution"])

    if flag:
        df.to_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'), header=True, mode='a', index=False)
    else:
        df.to_csv(os.path.join(path, 'Tests_SA_Congestion_Management.csv'), header=False, mode='a', index=False)

def main():
         
      
        start = time.time()
    
        mip, lst_vars = _create_mip_objective(P, T, TRAN, I, J, CP_COEF,
                                                    CS_COEF, CDIFF_COEF, V_P, V_S, P_0_T)
    
        mip = _mip_const_buckets(mip, T, P, TRAN, I, J)
        mip = _mip_const_power_balance(mip, P_0_T, P_0_P_T, T, P, I, V_P)
        mip = _mip_const_transformer(mip, S_MIN, S_MAX, V_S, TRAN, T, J)
        mip = _mip_const_branch(mip, BRANCH, I_0, S_P_B, S_TRAN_B, P_0_P_T, S_0_TRAN_T,
                                  I_B, V_P, V_S, T, P, TRAN, I, J)
        mip = _mip_const_producers_change(mip, P_MIN, P_MAX, V_P, P, T, I)
        mip = _mip_const_producers_time(mip, T_OFF, T_ON, T, P, I)
        print(mip.export_as_lp_string(),'\n')
    
        qubo, num_vars, hamiltonian, offset = _qubo_converter(mip, penalty=None)

        #print(qubo)

        bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)
        #sampler = SimulatedAnnealingSubproblemSampler(num_reads=10)

        sampler = LeapHybridSampler()

        sampleset = sampler.sample(bqm_binary, label='Example - CM')

  
        result_dwave = sampleset.first.sample

        stop = time.time()

        class C:
            pass
        result=C()

        setattr(result,"variables_dict",{})
        

        for idx, var in enumerate(qubo.variables_index):
            result.variables_dict[var]=result_dwave[idx]


        #result = sampler.sample(bqm_binary, label="example_qp", num_reads=1024)

        

        print(result)
  
        feasible = _check_result_(lst_vars, mip, result)
    
        final_time = stop - start
        #eigenvalue = result.fval
        solution = result.variables_dict
        execution_time = final_time
        parameters = None
        backend = 'Gurobi'
    
        _print_solution(solution, V_P, V_S)
    
        _write_csv(P, TRAN, T, BRANCH, I, J, CP_COEF, CS_COEF, CDIFF_COEF, V_P, V_S,
                      P_0_T, P_0_P_T, P_MIN, P_MAX, S_MIN, S_MAX, S_0_TRAN_T, I_0, I_B,
                      S_P_B, S_TRAN_B, T_ON, T_OFF, backend, feasible, final_time,
                      execution_time, 
                      #eigenvalue,
                       num_vars, parameters,
                      solution, path='.')
    
        print('\nThe algorithm with', num_vars, 'variables runs in (s):', final_time,'\n\n')

if __name__ == '__main__':
    
    main()

