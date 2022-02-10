# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:27:06 2021

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

from utilities import postprocessing as postprocessing_2d
from utilities import construct_A, preprocess_BO

from Problems.Facility_Location import centralised, f_4N_lowD, f_4N_highD, f_2N

np.random.seed(2) ; x_low = 0 ; y_low = 0 ; x_high = 5 ; y_high = 5   
data = {None: {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  
                  'N':   {None: 1},
                  'N_i': {None: 2},
                  'N_j': {None: 2},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18,},
                  'vf_k':  {1: 0.087,},
                  'mc_k':  {1: 250,},
                  'cv_k':  {1: 0.9,},
                  'ft_ik': {(1,1): 10, (2,1): 10,},
                  'ft_kj': {(1,1): 10, (1,2): 10,},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (2,1): 0.3,},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3,},
                  'D_ik_L': {(1,1): 0.5, (2,1): 0.5,},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5,},
                  'D_ik_U': {(1,1): 10, (2,1): 10,},
                  'D_kj_U': {(1,1): 10, (1,2): 10,},
                  'f_ik_U': {(1,1): 200, (2,1): 200,},
                  'f_kj_U': {(1,1): 200, (1,2): 200,},
                  'f_ik_L': {(1,1): 0, (2,1): 0,},
                  'f_kj_L': {(1,1): 0, (1,2): 0,},
                }
          }


res = centralised(data)
print(pyo.value(res.obj))

# raise ValueError('Error')

rho = 100000 # second
# rho = 1000 # first
N_it = 200

N = 4
N_var = 5

global_ind = list(np.arange(N_var)+1)
index_agents = global_ind
# x0 = np.array([pyo.value(res.x[i]) for i in res.x_k_s] + \
#               [pyo.value(res.f_k[k]) for k in res.k]   + \
#               [pyo.value(res.f_ik[1,1]), pyo.value(res.f_ik[1,2])] + \
#               [pyo.value(res.f_ik[2,1]), pyo.value(res.f_ik[2,2])] + \
#               [pyo.value(res.f_kj[1,1]), pyo.value(res.f_kj[2,1])] + \
#               [pyo.value(res.f_kj[1,2]), pyo.value(res.f_kj[2,2])])
# z = {i: [x0[i-1]] for i in global_ind}

# result = 0
# for idx in range(1,1+N):
#     res1 = f(z, rho, global_ind, index_agents, 'Supplier', idx) 
#     print('Objective supply', idx,  pyo.value(res1.obj))
#     for i in res1.x:
#         print('x'+str(i), pyo.value(res1.x[i]), 'z'+str(i), pyo.value(res1.z[i]))
#     res2 = f(z, rho, global_ind, index_agents, 'Market', idx)
#     print('Objective demand', idx,  pyo.value(res2.obj))
#     for i in res2.x:
#         print('x'+str(i), pyo.value(res2.x[i]), 'z'+str(i), pyo.value(res2.z[i]))
#     result += pyo.value(res1.obj) + pyo.value(res2.obj) 

x0 = np.array([2.5, 2.5] + \
              [100] + \
              [50] + \
              [50]) 

z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Supplier', 1, u_list = u_list, solver = solver, seed=seed)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Market', 1, u_list = u_list, solver = solver, seed=seed)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Supplier', 2, u_list = u_list, solver = solver, seed=seed)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Market', 2, u_list = u_list, solver = solver, seed=seed)

list_fi = [f1, f2, f3, f4]

ADMM_Scaled_system5d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system5d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system5d.solve_ADMM()


print('ADMM done')

bounds = np.array([[0, 5]]*2 + [[0, 350]]*3)
init_trust = 0.5
beta = 0.983

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Supplier', 1, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Market', 1, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Supplier', 2, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_lowD(z_list, rho, global_ind, index, 'Market', 2, u_list = u_list, solver = solver, seed=seed, bounds=bounds)

list_fi = [f1, f2, f3, f4]

x0_scaled = np.array([(x0[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x0))])

bounds_surr = np.array([[0, 1]]*5)

z = {i: x0_scaled[i-1] for i in global_ind}

Coordinator_ADMM_system5d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system5d.initialize_Decomp(rho, N_it, list_fi, z)
try:
    output_Coord1_5d = Coordinator_ADMM_system5d.solve(CUATRO, x0_scaled, bounds_surr, init_trust, 
                            budget = N_it, beta_red = beta)
except:
    output_Coord1_5d = {}
    z_dummy = {i: [x0_scaled[i-1]] for i in global_ind}
    y_dummy = float(np.sum([pyo.value(f(z_dummy, rho, global_ind, global_ind).obj) for f in list_fi]))
    output_Coord1_5d['f_best_so_far'] = np.zeros(N_it) + y_dummy
    output_Coord1_5d['samples_at_iteration'] = np.arange(1, N_it+1)
    output_Coord1_5d['x_best_so_fart'] = [x0_scaled]
print('Coord1 done')


A_dict = construct_A(index_agents, global_ind, N)
System_dataAL5d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL5d.initialize(rho, N_it, z, list_fi, A_dict)
try:
    System_dataAL5d.solve(6, init_trust, mu = 1e7, infeas_start = True)
except:
    print('Data-driven ALADIN failed')
    for ag in range(N):
        last_obj = System_dataAL5d.obj[ag+1][-1]
        N_dummy = len(System_dataAL5d.obj[ag+1])
        System_dataAL5d.obj[ag+1] += [last_obj]*(N_it-N_dummy)
print('Coord2 done')

def f_surr(x):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_pybobyqa5d = PyBobyqaWrapper().solve(f_surr, x0_scaled, bounds=bounds_surr.T, \
                                      maxfun= N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)
print('Py-BOBYQA done')    

def f_DIR(x, grad):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_DIRECT5d =  DIRECTWrapper().solve(f_DIR, x0_scaled, bounds_surr, \
                                    maxfun = N_it, constraints=1)

print('DIRECT done')

def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    z = {i: [x_temp[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi])
    

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0,1)} for i in range(len(x0_scaled))]
y0 = np.array([f_BO(x0_scaled)])
for j in range(len(FL_DIRECT5d['f_best_so_far'])):
    if FL_DIRECT5d['f_best_so_far'][j] > float(y0):
        FL_DIRECT5d['f_best_so_far'][j] = float(y0)


BO = BayesianOptimization(f=f_BO, domain=domain, X=x0_scaled.reshape((1,len(x0_scaled))), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post5d = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing_2d(ax1, ax2,  s, ADMM_Scaled_system5d, pyo.value(res.obj), c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing_2d(ax1, ax2, s, output_Coord1_5d, pyo.value(res.obj), coord_input = True, c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing_2d(ax1, ax2, s, FL_pybobyqa5d, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_2d(ax1, ax2, s, FL_DIRECT5d, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing_2d(ax1, ax2, s, System_dataAL5d, pyo.value(res.obj), ALADIN = True, c='darkviolet')
ax1, ax2 = out

s = 'BO'
out = postprocessing_2d(ax1, ax2, s, BO_post5d, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it

ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax1.set_xlabel('Number of function evaluations')
ax1.set_ylabel('Convergence')
ax1.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax1.legend()

ax2.set_xlabel('Number of function evaluations')
ax2.set_ylabel('Best function evaluation')
ax2.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax2.legend()

dim = len(x0)
problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")


pos_x = np.array([pyo.value(res.x[1])])
pos_y = np.array([pyo.value(res.x[2])])


fig1 = plt.figure() 
ax1 = fig1.add_subplot() 
s = 'Centralized' ; c = 'k'
ax1.scatter(pos_x, pos_y, marker='*', c = c,label=s)

x_all = output_Coord1_5d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'CUATRO1' ; c = 'darkorange'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_pybobyqa5d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'Py-BOBYQA' ; c = 'green'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_DIRECT5d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'DIRECT' ; c = 'red'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

s = 'CUATRO2' ; c = 'darkviolet'
x = np.array([System_dataAL5d.z_list[1][1][-1]])*np.array([bounds[0][1] - bounds[0][0]])
y = np.array([System_dataAL5d.z_list[1][2][-1]])*np.array([bounds[1][1] - bounds[1][0]])
ax1.scatter(x, y, c = c, label=s)

s = 'ADMM' ; c = 'dodgerblue'
x = np.array([ADMM_Scaled_system5d.z_list[1][-1]])
y = np.array([ADMM_Scaled_system5d.z_list[2][-1]])
ax1.scatter(x, y, c = c, label=s)

s = 'BO' ; c = 'saddlebrown'
x_best = BO.X[np.argmin(BO.Y)]
ax1.scatter(x_best[0], x_best[1], c = c, label=s)

supplier_x = [data[None]['x_i'][k] for k in range(1, 3)]
supplier_y = [data[None]['z_i'][k] for k in range(1, 3)]
demand_x = [data[None]['x_j'][k] for k in range(1, 3)]
demand_y = [data[None]['z_j'][k] for k in range(1, 3)]

plt.scatter(np.array(supplier_x), np.array(supplier_y), marker ='s', c = 'k', label = 'supply')
plt.scatter(np.array(demand_x), np.array(demand_y), marker='D', c = 'k', label = 'demand')
plt.legend()

problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_Vis.svg', format = "svg")

### higher dim


np.random.seed(2)

data = {None: {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  
                  'N':   {None: 2},
                  'N_i': {None: 2},
                  'N_j': {None: 2},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18, 2: 7.18},
                  'vf_k':  {1: 0.087, 2: 0.087},
                  'mc_k':  {1: 125, 2: 125},
                  'cv_k':  {1: 0.9, 2: 0.9},
                  'ft_ik': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'ft_kj': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'D_ik_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_ik_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'D_kj_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'f_ik_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_kj_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_ik_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                  'f_kj_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                }
          }






res = centralised(data)
print(pyo.value(res.obj))

# raise ValueError


# raise ValueError('Error')

# rho = 100000
N_it = 200

N = 4
N_var = 10

global_ind = list(np.arange(N_var)+1)
# index_agents = global_ind
# x0 = np.array([pyo.value(res.x[i]) for i in res.x_k_s] + \
#               [pyo.value(res.f_k[k]) for k in res.k]   + \
#               [pyo.value(res.f_ik[1,1]), pyo.value(res.f_ik[1,2])] + \
#               [pyo.value(res.f_ik[2,1]), pyo.value(res.f_ik[2,2])] + \
#               [pyo.value(res.f_kj[1,1]), pyo.value(res.f_kj[2,1])] + \
#               [pyo.value(res.f_kj[1,2]), pyo.value(res.f_kj[2,2])])
# z = {i: [x0[i-1]] for i in global_ind}

# result = 0
# for idx in range(1,1+N):
#     res1 = f(z, rho, global_ind, index_agents, 'Supplier', idx) 
#     print('Objective supply', idx,  pyo.value(res1.obj))
#     for i in res1.x:
#         print('x'+str(i), pyo.value(res1.x[i]), 'z'+str(i), pyo.value(res1.z[i]))
#     res2 = f(z, rho, global_ind, index_agents, 'Market', idx)
#     print('Objective demand', idx,  pyo.value(res2.obj))
#     for i in res2.x:
#         print('x'+str(i), pyo.value(res2.x[i]), 'z'+str(i), pyo.value(res2.z[i]))
#     result += pyo.value(res1.obj) + pyo.value(res2.obj) 

x0 = np.array([1.25, 2.5] + [3.75, 2.5] + \
              [100, 100] + \
              [50, 50] + \
              [50, 50]) 
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_highD(z_list, rho, global_ind, index, 'Supplier', 1, u_list = u_list, solver = solver, seed=seed)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_highD(z_list, rho, global_ind, index, 'Market', 1, u_list = u_list, solver = solver, seed=seed)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_highD(z_list, rho, global_ind, index, 'Supplier', 2, u_list = u_list, solver = solver, seed=seed)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_4N_highD(z_list, rho, global_ind, index, 'Market', 2, u_list = u_list, solver = solver, seed=seed)

list_fi = [f1, f2, f3, f4]

ADMM_Scaled_system10d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system10d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system10d.solve_ADMM()


print('ADMM done')

bounds = np.array([[0, 5]]*4 + [[0, 150]]*6)
init_trust = 0.5
beta = 0.983

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_highD(z_list, rho, global_ind, index, 'Supplier', 1, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_highD(z_list, rho, global_ind, index, 'Market', 1, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_highD(z_list, rho, global_ind, index, 'Supplier', 2, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_4N_highD(z_list, rho, global_ind, index, 'Market', 2, u_list = u_list, solver = solver, seed=seed, bounds=bounds)

list_fi = [f1, f2, f3, f4]

x0_scaled = np.array([(x0[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x0))])

bounds_surr = np.array([[0, 1]]*10)

z = {i: x0_scaled[i-1] for i in global_ind}

Coordinator_ADMM_system10d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system10d.initialize_Decomp(rho, N_it, list_fi, z)
try:
    output_Coord1_10d = Coordinator_ADMM_system10d.solve(CUATRO, x0_scaled, bounds_surr, init_trust, 
                            budget = N_it, beta_red = beta)
except:
    output_Coord1_10d = {}
    z_dummy = {i: [x0_scaled[i-1]] for i in global_ind}
    y_dummy = float(np.sum([pyo.value(f(z_dummy, rho, global_ind, global_ind).obj) for f in list_fi]))
    output_Coord1_10d['f_best_so_far'] = np.zeros(N_it) + y_dummy
    output_Coord1_10d['samples_at_iteration'] = np.arange(1, N_it+1)
    output_Coord1_10d['x_best_so_fart'] = [x0_scaled]
print('Coord1 done')


A_dict = construct_A(index_agents, global_ind, N)
System_dataAL10d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL10d.initialize(rho, N_it, z, list_fi, A_dict)
try:
    System_dataAL10d.solve(6, init_trust, mu = 1e7, infeas_start = True)
except:
    print('Data-driven ALADIN failed')
    for ag in range(N):
        last_obj = System_dataAL10d.obj[ag+1][-1]
        N_dummy = len(System_dataAL10d.obj[ag+1])
        System_dataAL10d.obj[ag+1] += [last_obj]*(N_it-N_dummy)
print('Coord2 done')

def f_surr(x):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_pybobyqa10d = PyBobyqaWrapper().solve(f_surr, x0_scaled, bounds=bounds_surr.T, \
                                      maxfun= N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)
print('Py-BOBYQA done')    

def f_DIR(x, grad):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_DIRECT10d =  DIRECTWrapper().solve(f_DIR, x0_scaled, bounds_surr, \
                                    maxfun = N_it, constraints=1)
print('DIRECT done')


def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    z = {i: [x_temp[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi])
    

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0,1)} for i in range(len(x0_scaled))]
y0 = np.array([f_BO(x0_scaled)])
for j in range(len(FL_DIRECT5d['f_best_so_far'])):
    if FL_DIRECT10d['f_best_so_far'][j] > float(y0):
        FL_DIRECT10d['f_best_so_far'][j] = float(y0)

BO = BayesianOptimization(f=f_BO, domain=domain, X=x0_scaled.reshape((1,len(x0_scaled))), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post10d = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing_2d(ax1, ax2,  s, ADMM_Scaled_system10d, pyo.value(res.obj), c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing_2d(ax1, ax2, s, output_Coord1_10d, pyo.value(res.obj), coord_input = True, c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing_2d(ax1, ax2, s, FL_pybobyqa10d, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_2d(ax1, ax2, s, FL_DIRECT10d, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing_2d(ax1, ax2, s, System_dataAL10d, pyo.value(res.obj), ALADIN = True, c='darkviolet')
ax1, ax2 = out

s = 'BO'
out = postprocessing_2d(ax1, ax2, s, BO_post10d, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it

ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax1.set_xlabel('Number of function evaluations')
ax1.set_ylabel('Convergence')
ax1.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax1.legend()

ax2.set_xlabel('Number of function evaluations')
ax2.set_ylabel('Best function evaluation')
ax2.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax2.legend()

dim = len(x0)
problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")

pos_x = np.array([pyo.value(res.x[1]), pyo.value(res.x[2])])
pos_y = np.array([pyo.value(res.x[3]), pyo.value(res.x[4])])

fig1 = plt.figure() 
ax1 = fig1.add_subplot() 
s = 'Centralized' ; c = 'k'
ax1.scatter(pos_x, pos_y, marker='*', c = c, label=s)

x_all = output_Coord1_10d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'CUATRO1' ; c = 'darkorange'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_pybobyqa10d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'Py-BOBYQA' ; c = 'green'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_DIRECT10d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'DIRECT' ; c = 'red'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

s = 'CUATRO2' ; c = 'darkviolet'
x_all = np.array([System_dataAL10d.z_list[1][i][-1]*(bounds[i-1][1] - bounds[i-1][0]) for i in global_ind])
x = x_all[:2] ; y = x_all[2:4]
ax1.scatter(x, y, c = c, label=s)

s = 'ADMM' ; c = 'dodgerblue'
x = np.array([ADMM_Scaled_system10d.z_list[1][-1], ADMM_Scaled_system10d.z_list[2][-1]])
y = np.array([ADMM_Scaled_system10d.z_list[3][-1], ADMM_Scaled_system10d.z_list[4][-1]])
ax1.scatter(x, y, c = c, label=s)

s = 'BO' ; c = 'saddlebrown'
x_all = BO.X[np.argmin(BO.Y)]
x = x_all[:2] ; y = x_all[2:4]
ax1.scatter(x, y, c = c, label=s)

supplier_x = [data[None]['x_i'][k] for k in range(1, 3)]
supplier_y = [data[None]['z_i'][k] for k in range(1, 3)]
demand_x = [data[None]['x_j'][k] for k in range(1, 3)]
demand_y = [data[None]['z_j'][k] for k in range(1, 3)]

plt.scatter(np.array(supplier_x), np.array(supplier_y), marker ='s', c = 'k', label = 'supply')
plt.scatter(np.array(demand_x), np.array(demand_y), marker='D', c = 'k', label = 'demand')
plt.legend()


problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_Vis.svg', format = "svg")


### 2N
### low dim


np.random.seed(2) ; x_low = 0 ; y_low = 0 ; x_high = 5 ; y_high = 5   
data = {None: {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  
                  'N':   {None: 1},
                  'N_i': {None: 2},
                  'N_j': {None: 2},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18,},
                  'vf_k':  {1: 0.087,},
                  'mc_k':  {1: 250,},
                  'cv_k':  {1: 0.9,},
                  'ft_ik': {(1,1): 10, (2,1): 10,},
                  'ft_kj': {(1,1): 10, (1,2): 10,},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (2,1): 0.3,},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3,},
                  'D_ik_L': {(1,1): 0.5, (2,1): 0.5,},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5,},
                  'D_ik_U': {(1,1): 10, (2,1): 10,},
                  'D_kj_U': {(1,1): 10, (1,2): 10,},
                  'f_ik_U': {(1,1): 200, (2,1): 200,},
                  'f_kj_U': {(1,1): 200, (1,2): 200,},
                  'f_ik_L': {(1,1): 0, (2,1): 0,},
                  'f_kj_L': {(1,1): 0, (1,2): 0,},
                }
          }



res = centralised(data)
print(pyo.value(res.obj))

# raise ValueError('Error')

rho = 1e5 # Second
# rho = 1e3 # First
N_it = 100

N = 2
N_var = 3

global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

x0 = np.array([2.5, 2.5, 100])
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}
def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_2N(z_list, rho, global_ind, index, 'Supplier', data, u_list = u_list, solver = solver, seed=seed)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_2N(z_list, rho, global_ind, index, 'Market', data, u_list = u_list, solver = solver, seed=seed)

list_fi = [f1, f2]

ADMM_Scaled_system3d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system3d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system3d.solve_ADMM()


print('ADMM done')

bounds = np.array([[0, 5]]*2 + [[0, 350]])
init_trust = 0.5
beta = 0.983

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_2N(z_list, rho, global_ind, index, 'Supplier', data, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_2N(z_list, rho, global_ind, index, 'Market', data, u_list = u_list, solver = solver, seed=seed, bounds=bounds)

list_fi = [f1, f2]

x0_scaled = np.array([(x0[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x0))])

bounds_surr = np.array([[0, 1]]*N_var)

z = {i: x0_scaled[i-1] for i in global_ind}


Coordinator_ADMM_system3d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system3d.initialize_Decomp(rho, N_it, list_fi, z)
try:
    output_Coord1_3d = Coordinator_ADMM_system3d.solve(CUATRO, x0_scaled, bounds_surr, init_trust, 
                            budget = N_it, beta_red = beta)
except:
    output_Coord1_3d = {}
    z_dummy = {i: [x0_scaled[i-1]] for i in global_ind}
    y_dummy = float(np.sum([pyo.value(f(z_dummy, rho, global_ind, global_ind).obj) for f in list_fi]))
    output_Coord1_3d['f_best_so_far'] = np.zeros(N_it) + y_dummy
    output_Coord1_3d['samples_at_iteration'] = np.arange(1, N_it+1)
    output_Coord1_3d['x_best_so_fart'] = [x0_scaled]
print('Coord1 done')


A_dict = construct_A(index_agents, global_ind, N)
System_dataAL3d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL3d.initialize(rho, N_it, z, list_fi, A_dict)
try:
    System_dataAL3d.solve(6, init_trust, mu = 1e7, infeas_start = True)
except:
    print('Data-driven ALADIN failed')
    for ag in range(N):
        last_obj = System_dataAL3d.obj[ag+1][-1]
        N_dummy = len(System_dataAL3d.obj[ag+1])
        System_dataAL3d.obj[ag+1] += [last_obj]*(N_it-N_dummy)
print('Coord2 done')

def f_surr(x):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_pybobyqa3d = PyBobyqaWrapper().solve(f_surr, x0_scaled, bounds=bounds_surr.T, \
                                      maxfun= N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)
print('Py-BOBYQA done')    

def f_DIR(x, grad):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_DIRECT3d =  DIRECTWrapper().solve(f_DIR, x0_scaled, bounds_surr, \
                                    maxfun = N_it, constraints=1)

print('DIRECT done')

def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    z = {i: [x_temp[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi])
    

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0,1)} for i in range(len(x0))]
y0 = np.array([f_BO(x0_scaled)])
for j in range(len(FL_DIRECT3d['f_best_so_far'])):
    if FL_DIRECT3d['f_best_so_far'][j] > float(y0):
        FL_DIRECT3d['f_best_so_far'][j] = float(y0)

BO = BayesianOptimization(f=f_BO, domain=domain, X=x0_scaled.reshape((1,len(x0_scaled))), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post3d = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing_2d(ax1, ax2,  s, ADMM_Scaled_system3d, pyo.value(res.obj), c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing_2d(ax1, ax2, s, output_Coord1_3d, pyo.value(res.obj), coord_input = True, c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing_2d(ax1, ax2, s, FL_pybobyqa3d, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_2d(ax1, ax2, s, FL_DIRECT3d, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing_2d(ax1, ax2, s, System_dataAL3d, pyo.value(res.obj), ALADIN = True, c='darkviolet')
ax1, ax2 = out

s = 'BO'
out = postprocessing_2d(ax1, ax2, s, BO_post3d, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it

ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax1.set_xlabel('Number of function evaluations')
ax1.set_ylabel('Convergence')
ax1.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax1.legend()

ax2.set_xlabel('Number of function evaluations')
ax2.set_ylabel('Best function evaluation')
ax2.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax2.legend()


dim = len(x0)
problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")


pos_x = np.array([pyo.value(res.x[1])])
pos_y = np.array([pyo.value(res.x[2])])

fig1 = plt.figure() 
ax1 = fig1.add_subplot() 
s = 'Centralized' ; c = 'k'
ax1.scatter(pos_x, pos_y, marker='*', c = c,label=s)

x_all = output_Coord1_3d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'CUATRO1' ; c = 'darkorange'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_pybobyqa3d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'Py-BOBYQA' ; c = 'green'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_DIRECT3d['x_best_so_far'][-1][:2]*(np.array([bounds[i][1] - bounds[i][0] for i in range(2)]))
s = 'DIRECT' ; c = 'red'
x = x_all[:1] ; y = x_all[1:]
ax1.scatter(x, y, c = c, label=s)

s = 'CUATRO2' ; c = 'darkviolet'
x = np.array([System_dataAL3d.z_list[1][1][-1]])*np.array([bounds[0][1] - bounds[0][0]])
y = np.array([System_dataAL3d.z_list[1][2][-1]])*np.array([bounds[1][1] - bounds[1][0]])
ax1.scatter(x, y, c = c, label=s)

s = 'ADMM' ; c = 'dodgerblue'
x = np.array([ADMM_Scaled_system3d.z_list[1][-1]])
y = np.array([ADMM_Scaled_system3d.z_list[2][-1]])
ax1.scatter(x, y, c = c, label=s)

s = 'BO' ; c = 'saddlebrown'
x_best = BO.X[np.argmin(BO.Y)]
ax1.scatter(x_best[0], x_best[1], c = c, label=s)

supplier_x = [data[None]['x_i'][k] for k in range(1, 3)]
supplier_y = [data[None]['z_i'][k] for k in range(1, 3)]
demand_x = [data[None]['x_j'][k] for k in range(1, 3)]
demand_y = [data[None]['z_j'][k] for k in range(1, 3)]

plt.scatter(np.array(supplier_x), np.array(supplier_y), marker ='s', c = 'k', label = 'supply')
plt.scatter(np.array(demand_x), np.array(demand_y), marker='D', c = 'k', label = 'demand')
plt.legend()

problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_Vis.svg', format = "svg")


np.random.seed(2)

data = {None: {
                  'x_i': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)}, 
                  'x_j': {1: np.random.uniform(low=x_low, high=x_high), 
                          2: np.random.uniform(low=x_low, high=x_high)},
                  'z_i': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)}, 
                  'z_j': {1: np.random.uniform(low=y_low, high=y_high), 
                          2: np.random.uniform(low=y_low, high=y_high)},
                  
                  'N':   {None: 2},
                  'N_i': {None: 2},
                  'N_j': {None: 2},
                  'cs_i':  {1: 20, 2: 22},
                  'a_i':   {1: 120, 2: 120},
                  'd_j':   {1: 100, 2: 100},
                  'ff_k':  {1: 7.18, 2: 7.18},
                  'vf_k':  {1: 0.087, 2: 0.087},
                  'mc_k':  {1: 125, 2: 125},
                  'cv_k':  {1: 0.9, 2: 0.9},
                  'ft_ik': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'ft_kj': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  # 'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  # 'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_ik': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'vt_kj': {(1,1): 0.3, (1,2): 0.3, (2,1): 0.3, (2,2): 0.3},
                  'D_ik_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_kj_L': {(1,1): 0.5, (1,2): 0.5, (2,1): 0.5, (2,2): 0.5},
                  'D_ik_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'D_kj_U': {(1,1): 10, (1,2): 10, (2,1): 10, (2,2): 10},
                  'f_ik_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_kj_U': {(1,1): 200, (1,2): 200, (2,1): 200, (2,2): 200},
                  'f_ik_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                  'f_kj_L': {(1,1): 0, (1,2): 0, (2,1): 0, (2,2): 0},
                }
          }


res = centralised(data)
print(pyo.value(res.obj))

# raise ValueError


# raise ValueError('Error')

# rho = 100000
N_it = 100

N = 2
N_var = 6

global_ind = list(np.arange(N_var)+1)
# index_agents = global_ind
# x0 = np.array([pyo.value(res.x[i]) for i in res.x_k_s] + \
#               [pyo.value(res.f_k[k]) for k in res.k]   + \
#               [pyo.value(res.f_ik[1,1]), pyo.value(res.f_ik[1,2])] + \
#               [pyo.value(res.f_ik[2,1]), pyo.value(res.f_ik[2,2])] + \
#               [pyo.value(res.f_kj[1,1]), pyo.value(res.f_kj[2,1])] + \
#               [pyo.value(res.f_kj[1,2]), pyo.value(res.f_kj[2,2])])
# z = {i: [x0[i-1]] for i in global_ind}

# result = 0
# for idx in range(1,1+N):
#     res1 = f(z, rho, global_ind, index_agents, 'Supplier', idx) 
#     print('Objective supply', idx,  pyo.value(res1.obj))
#     for i in res1.x:
#         print('x'+str(i), pyo.value(res1.x[i]), 'z'+str(i), pyo.value(res1.z[i]))
#     res2 = f(z, rho, global_ind, index_agents, 'Market', idx)
#     print('Objective demand', idx,  pyo.value(res2.obj))
#     for i in res2.x:
#         print('x'+str(i), pyo.value(res2.x[i]), 'z'+str(i), pyo.value(res2.z[i]))
#     result += pyo.value(res1.obj) + pyo.value(res2.obj) 

x0 = np.array([1.25, 2.5] + [3.75, 2.5] + \
              [100, 100]) 
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_2N(z_list, rho, global_ind, index, 'Supplier', data, u_list = u_list, solver = solver, seed=seed)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2):
    return f_2N(z_list, rho, global_ind, index, 'Market', data, u_list = u_list, solver = solver, seed=seed)

list_fi = [f1, f2]

ADMM_Scaled_system6d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system6d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system6d.solve_ADMM()


print('ADMM done')

bounds = np.array([[0, 5]]*2 + [[0, 150]]*4)
init_trust = 0.5
beta = 0.983

def f1(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_2N(z_list, rho, global_ind, index, 'Supplier', data, u_list = u_list, solver = solver, seed=seed, bounds=bounds)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False, seed=2, bounds=bounds):
    return f_2N(z_list, rho, global_ind, index, 'Market', data, u_list = u_list, solver = solver, seed=seed, bounds=bounds)

list_fi = [f1, f2]

x0_scaled = np.array([(x0[i] - bounds[i][0])/(bounds[i][1]-bounds[i][0]) for i in range(len(x0))])

bounds_surr = np.array([[0, 1]]*N_var)

z = {i: x0_scaled[i-1] for i in global_ind}

Coordinator_ADMM_system6d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system6d.initialize_Decomp(rho, N_it, list_fi, z)
try:
    output_Coord1_6d = Coordinator_ADMM_system5d.solve(CUATRO, x0_scaled, bounds_surr, init_trust, 
                            budget = N_it, beta_red = beta)
except:
    output_Coord1_6d = {}
    z_dummy = {i: [x0_scaled[i-1]] for i in global_ind}
    y_dummy = float(np.sum([pyo.value(f(z_dummy, rho, global_ind, global_ind).obj) for f in list_fi]))
    output_Coord1_6d['f_best_so_far'] = np.zeros(N_it) + y_dummy
    output_Coord1_6d['samples_at_iteration'] = np.arange(1, N_it+1)
    output_Coord1_6d['x_best_so_fart'] = [x0_scaled]
print('Coord1 done')


A_dict = construct_A(index_agents, global_ind, N)
System_dataAL6d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL6d.initialize(rho, N_it, z, list_fi, A_dict)
try:
    System_dataAL6d.solve(6, init_trust, mu = 1e7, infeas_start = True)
except:
    print('Data-driven ALADIN failed')
    for ag in range(N):
        last_obj = System_dataAL6d.obj[ag+1][-1]
        N_dummy = len(System_dataAL6d.obj[ag+1])
        System_dataAL6d.obj[ag+1] += [last_obj]*(N_it-N_dummy)
print('Coord2 done')

def f_surr(x):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_pybobyqa6d = PyBobyqaWrapper().solve(f_surr, x0_scaled, bounds=bounds_surr.T, \
                                      maxfun= N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)
print('Py-BOBYQA done')    

def f_DIR(x, grad):
    z = {i: [x[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi]), [0]

FL_DIRECT6d =  DIRECTWrapper().solve(f_DIR, x0_scaled, bounds_surr, \
                                    maxfun = N_it, constraints=1)

print('DIRECT done')

def f_BO(x):
    if x.ndim > 1:
       x_temp = x[-1] 
    else:
       x_temp = x
    z = {i: [x_temp[i-1]] for i in global_ind}
    iterables = [rho, global_ind, global_ind]
    return np.sum([pyo.value(f(z, *iterables).obj) for f in list_fi])
    

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0,1)} for i in range(len(x0))]
y0 = np.array([f_BO(x0_scaled)])
for j in range(len(FL_DIRECT6d['f_best_so_far'])):
    if FL_DIRECT6d['f_best_so_far'][j] > float(y0):
        FL_DIRECT6d['f_best_so_far'][j] = float(y0)


BO = BayesianOptimization(f=f_BO, domain=domain, X=x0_scaled.reshape((1,len(x0_scaled))), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post6d = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing_2d(ax1, ax2,  s, ADMM_Scaled_system6d, pyo.value(res.obj), c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing_2d(ax1, ax2, s, output_Coord1_6d, pyo.value(res.obj), coord_input = True, c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing_2d(ax1, ax2, s, FL_pybobyqa6d, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_2d(ax1, ax2, s, FL_DIRECT6d, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing_2d(ax1, ax2, s, System_dataAL6d, pyo.value(res.obj), ALADIN = True, c='darkviolet')
ax1, ax2 = out

s = 'BO'
out = postprocessing_2d(ax1, ax2, s, BO_post6d, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it

ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')

# ax1.scatter
# ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
ax1.set_xlabel('Number of function evaluations')
ax1.set_ylabel('Convergence')
ax1.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax1.legend()

ax2.set_xlabel('Number of function evaluations')
ax2.set_ylabel('Best function evaluation')
ax2.set_yscale('log')
# ax1.set_ylim([5e3, 7e3])
ax2.legend()


dim = len(x0)
problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")



pos_x = np.array([pyo.value(res.x[1]), pyo.value(res.x[2])])
pos_y = np.array([pyo.value(res.x[3]), pyo.value(res.x[4])])

fig1 = plt.figure() 
ax1 = fig1.add_subplot() 
s = 'Centralized' ; c = 'k'
ax1.scatter(pos_x, pos_y, marker='*', c = c, label=s)

x_all = output_Coord1_6d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'CUATRO1' ; c = 'darkorange'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_pybobyqa6d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'Py-BOBYQA' ; c = 'green'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

x_all = FL_DIRECT6d['x_best_so_far'][-1][:4]*(np.array([bounds[i][1] - bounds[i][0] for i in range(4)]))
s = 'DIRECT' ; c = 'red'
x = x_all[:2] ; y = x_all[2:]
ax1.scatter(x, y, c = c, label=s)

s = 'CUATRO2' ; c = 'darkviolet'
x_all = np.array([System_dataAL6d.z_list[1][i][-1]*(bounds[i-1][1] - bounds[i-1][0]) for i in global_ind])
x = x_all[:2] ; y = x_all[2:4]
ax1.scatter(x, y, c = c, label=s)

s = 'ADMM' ; c = 'dodgerblue'
x = np.array([ADMM_Scaled_system6d.z_list[1][-1], ADMM_Scaled_system6d.z_list[2][-1]])
y = np.array([ADMM_Scaled_system6d.z_list[3][-1], ADMM_Scaled_system6d.z_list[4][-1]])
ax1.scatter(x, y, c = c, label=s)

s = 'BO' ; c = 'saddlebrown'
x_all = BO.X[np.argmin(BO.Y)]
x = x_all[:2] ; y = x_all[2:4]
ax1.scatter(x, y, c = c, label=s)

supplier_x = [data[None]['x_i'][k] for k in range(1, 3)]
supplier_y = [data[None]['z_i'][k] for k in range(1, 3)]
demand_x = [data[None]['x_j'][k] for k in range(1, 3)]
demand_y = [data[None]['z_j'][k] for k in range(1, 3)]

plt.scatter(np.array(supplier_x), np.array(supplier_y), marker ='s', c = 'k', label = 'supply')
plt.scatter(np.array(demand_x), np.array(demand_y), marker='D', c = 'k', label = 'demand')
plt.legend()


problem = 'Facility_Location_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_Vis.svg', format = "svg")









