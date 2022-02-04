# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:33:43 2021

@author: dv516
"""

from Problems.ToyProblem2 import f1, f2, f3, f4

from Algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from Algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper
from Algorithms.ALADIN_Data import System as ALADIN_Data
from Algorithms.ADMM_Scaled_Consensus import System as ADMM_Scaled
# from Algorithms.Coordinator_Augmented import System as Coordinator_ADMM
from Algorithms.Coordinator_explConstr import System as Coordinator_withConstr
from Algorithms.CUATRO import CUATRO
from GPyOpt.methods import BayesianOptimization

import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from utilities import postprocessing, preprocess_BO


rho = 5000
N_it = 50

N = 4
N_var = 10
list_fi = [f1, f2, f3, f4]

global_ind = [1, 2]
index_agents = {1: [1, 2, 3, 4], 2: [1, 2, 5, 6], 
                3: [1, 2, 7, 8], 4: [1, 2, 9, 10]}
z = {1: 2, 2:5}
dim = len(z)

actual_f = -20.69379
actual_x = [2.171996353, 2.363683]



ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

s = 'ADMM_Scaled'
print(s+': ', 'Done')

# rho = 5000 # previous run was 500 # originally 5000

x0 = np.array([z[1], z[2]])
bounds = np.array([[-10, 10], [-10, 10]])
init_trust = 1
beta = 0.5

Coordinator_withConstr_system = Coordinator_withConstr(N, N_var, index_agents, global_ind)
Coordinator_withConstr_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord = Coordinator_withConstr_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)    

s = 'Coordinator'
print(s+': ', 'Done')


A1, A2, A3, A4 = np.zeros((6, 2)), np.zeros((6, 2)), np.zeros((6, 2)), np.zeros((6, 2))
A1[0][0] = 1 ; A1[3][1] = 1
A2[0][0] = -1 ; A2[1][0] = 1 ; A2[3][1] = -1 ; A2[4][1] = 1
A3[1][0] = -1 ; A3[2][0] = 1 ; A3[4][1] = -1 ; A3[5][1] = 1
A4[2][0] = -1 ; A4[5][1] = -1
A_dict = {1: A1, 2: A2, 3: A3, 4: A4}


System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(6, init_trust, mu = 1e7)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {global_ind[i]: [x[i]] for i in range(dim)}
    return np.sum([pyo.value(list_fi[i](z_list, rho, global_ind, index_agents[i+1]).obj) for i in range(N)]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {global_ind[i]: [x_temp[i]] for i in range(dim)}
    return np.sum([pyo.value(list_fi[i](z_list, rho, global_ind, index_agents[i+1]).obj) for i in range(N)])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-10,10)} for i in range(dim)]

y0 = np.array([f_BO(x0)])
for i in range(len(DIRECT['f_best_so_far'])):
    if DIRECT['f_best_so_far'][i] > float(y0):
        DIRECT['f_best_so_far'][i] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)


fig1 = plt.figure() 
ax1 = fig1.add_subplot() 
fig2 = plt.figure() 
ax2 = fig2.add_subplot() 

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, actual_f)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord, actual_f, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, actual_f, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, actual_f, coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, actual_f, ALADIN = True, init=float(y0))
ax1, ax2 = out

s = 'BO'
out = postprocessing(ax1, ax2, s, BO_post, actual_f, BO = True)
ax1, ax2 = out

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax1.set_xlabel('Number of function evaluations')
ax1.set_ylabel('Convergence')
ax1.set_yscale('log')
ax1.legend()

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax2.set_xlabel('Number of function evaluations')
ax2.set_ylabel('Best function evaluation')
# ax2.set_yscale('log')
ax2.plot([1, N_it], [actual_f, actual_f], '--k', label = 'Centralized')
ax2.legend()

problem = 'Test_function_2'
fig1.savefig('../Figures/' + problem +'_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem +'_evals.svg', format = "svg")

