# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:33:43 2021

@author: dv516
"""

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
# from Algorithms.ADMM_Scaled_Consensus_acc import System as ADMM_Scaled_acc
from Algorithms.ADMM_Scaled_Consensus import System as ADMM_Scaled
from Algorithms.ALADIN_Data import System as ALADIN_Data

from Algorithms.Coordinator_Augmented import System as Coordinator_ADMM
from Algorithms.CUATRO import CUATRO

from Algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from Algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper
from GPyOpt.methods import BayesianOptimization

from utilities import construct_A, preprocess_BO, postprocessing

from Problems.MAC import f1 as f1Raw
from Problems.MAC import f2 as f2Raw
from Problems.MAC import f1Lin as f1LinRaw
from Problems.MAC import f2Lin as f2LinRaw

rho = 5000
N_it = 1000

N = 2
N_var = 10

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1, f2]
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}

actual_f = 0
# actual_x = 0.398


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, 0)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, 0, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, 0, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, 0, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, 0, ALADIN = True)
ax1, ax2 = out

s = 'BO'
out = postprocessing(ax1, ax2, s, BO_post, 0, BO = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_nonconv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")


def f1Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1Lin, f2Lin]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, actual_f)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, actual_f, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, actual_f, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, actual_f, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, actual_f, ALADIN = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_conv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")


N_var = 5

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1, f2]
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}

actual_f = 0
# actual_x = 0.398


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, 0)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, 0, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, 0, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, 0, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, 0, ALADIN = True)
ax1, ax2 = out

s = 'BO'
out = postprocessing(ax1, ax2, s, BO_post, 0, BO = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_nonconv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")


def f1Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1Lin, f2Lin]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, actual_f)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, actual_f, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, actual_f, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, actual_f, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, actual_f, ALADIN = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_conv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")


N_var = 25

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2Raw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1, f2]
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}

actual_f = 0
# actual_x = 0.398


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, 0)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, 0, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, 0, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, 0, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, 0, ALADIN = True)
ax1, ax2 = out

s = 'BO'
out = postprocessing(ax1, ax2, s, BO_post, 0, BO = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_nonconv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")


def f1Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f1LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)
def f2Lin(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f2LinRaw(z_list, rho, global_ind, index, u_list = None, solver = False, dim=N_var)

list_fi = [f1Lin, f2Lin]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
global_ind = [i+1 for i in range(N_var)]
index_agents = {1: global_ind, 2: global_ind}
z = {n: 1/N_var for n in global_ind}


ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)    

s = 'Coordinator_ADMM'
print(s+': ', 'Done')


A_dict = construct_A(index_agents, global_ind, N, only_global = True)

System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)

s = 'ALADIN_Data'
print(s+': ', 'Done')


def f_pbqa(x):
    z_list = {i: [x[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
f_DIR = lambda x, grad: f_pbqa(x)
def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    # temp_dict = {i+1: x[:,i] for i in range(len(x))}
    z_list = {i: [x_temp[i-1]] for i in global_ind}
    return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

pybobyqa = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
DIRECT['f_best_so_far'][0] = float(y0)
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,N_var)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, actual_f)
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, actual_f, coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, actual_f, coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, actual_f, coord_input = True, init=float(y0))
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, actual_f, ALADIN = True)
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
ax2.set_yscale('log')
# ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_conv_'
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")







