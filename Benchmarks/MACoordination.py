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

from utilities import construct_A, preprocess_BO, postprocessing, postprocess_ADMM
from utilities import postprocessing_List

from Problems.MAC import f1 as f1Raw
from Problems.MAC import f2 as f2Raw
from Problems.MAC import f1Lin as f1LinRaw
from Problems.MAC import f2Lin as f2LinRaw
from Problems.MAC import centralized_conv, centralized_nonconv

import pickle

rho = 5000
N_it = 500
N_runs = 5

N = 2
N_var = 10

save_data_list = []

data_centr = {None: {'I': {None: list(np.arange(N_var)+1)}}}
res = centralized_nonconv(data_centr, dim=N_var)

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


ADMM_Scaled_system10nonconv = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system10nonconv.initialize_ADMM(rho/10, N_it, list_fi, z)
ADMM_Scaled_system10nonconv.solve_ADMM()

save_data_list += [postprocess_ADMM(ADMM_Scaled_system10nonconv)]

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

ADMM_CUATRO_list10nonconv = []
s = 'ADMM_CUATRO'
for i in range(1):
    Coordinator_ADMM_system10nonconv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
    Coordinator_ADMM_system10nonconv.initialize_Decomp(rho, N_it, list_fi, z)
    output_Coord1_10nonconv = Coordinator_ADMM_system10nonconv.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)  
    ADMM_CUATRO_list10nonconv += [output_Coord1_10nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord1 done')

save_data_list += [ADMM_CUATRO_list10nonconv]


A_dict = construct_A(index_agents, global_ind, N, only_global = True)
ALADIN_CUATRO_list10nonconv = []
s = 'ALADIN_CUATRO'
for i in range(1):
    System_dataAL10nonconv = ALADIN_Data(N, N_var, index_agents, global_ind)
    System_dataAL10nonconv.initialize(rho, N_it, z, list_fi, A_dict)
    System_dataAL10nonconv.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)
    ALADIN_CUATRO_list10nonconv += [System_dataAL10nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord2 done')

save_data_list += [ALADIN_CUATRO_list10nonconv]

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

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
y0 = np.array([f_BO(x0)])

pybobyqa10nonconv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

save_data_list += [pybobyqa10nonconv]
    
s = 'DIRECT'
DIRECT_List10nonconv = []
for i in range(N_runs): 
    DIRECT10nonconv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                    maxfun = N_it, constraints=1)
    for j in range(len(DIRECT10nonconv['f_best_so_far'])):
        if DIRECT10nonconv['f_best_so_far'][j] > float(y0):
            DIRECT10nonconv['f_best_so_far'][j] = float(y0)
    DIRECT_List10nonconv += [DIRECT10nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('DIRECT done')

save_data_list += [DIRECT_List10nonconv]

s = 'BO'
BO_List10nonconv = []
for i in range(N_runs):
    BO10nonconv = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,len(x0))), Y=y0.reshape((1,1)))
    BO10nonconv.run_optimization(max_iter=N_it, eps=0)
    BO_post10nonconv = preprocess_BO(BO10nonconv.Y.flatten(), y0, N_eval=N_it)
    BO_List10nonconv += [BO_post10nonconv]
    print(s + ' run ' + str(i+1) + ': Done')

save_data_list += [BO_List10nonconv]    

s_list = ['ADMM', 'ADMM_CUATRO', 'ALADIN_CUATRO', 'Py-BOBYQA', 
          'DIRECT-L', 'GPyOpt']

dim = len(x0)
problem = 'MAC_nonconv_'

for k in range(len(s_list)):
    with open('../Data/' + problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'wb') as handle:
        pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 


data_dict = {}
for k in range(len(s_list)):
    with open('../Data/'+ problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'rb') as handle:
        data_dict[s_list[k]] = pickle.load(handle)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM'
out = postprocessing(ax1, ax2,  s, data_dict[s], pyo.value(res.obj), c='dodgerblue', N=N)
ax1, ax2 = out

s = 'ADMM_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='darkorange', N_it=N_it)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='red', N_it=N_it)
ax1, ax2 = out

s = 'ALADIN_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), ALADIN = True, c='darkviolet', N_it=N_it)
ax1, ax2 = out

s = 'GPyOpt'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), BO = True, c='saddlebrown', N_it=N_it)
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
ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_nonconv_'
fig1.savefig('../Figures/' + problem + str(N_var) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")

data_centr = {None: {'I': {None: list(np.arange(N_var)+1)}}}
res = centralized_conv(data_centr, dim=N_var)

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

save_data_list = []

ADMM_Scaled_system10conv = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system10conv.initialize_ADMM(rho/10, N_it, list_fi, z)
ADMM_Scaled_system10conv.solve_ADMM()

save_data_list += [postprocess_ADMM(ADMM_Scaled_system10conv)]

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95


ADMM_CUATRO_list10conv = []
s = 'ADMM_CUATRO'
for i in range(1):
    Coordinator_ADMM_system10conv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
    Coordinator_ADMM_system10conv.initialize_Decomp(rho, N_it, list_fi, z)
    output_Coord1_10conv = Coordinator_ADMM_system10conv.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)  
    ADMM_CUATRO_list10conv += [output_Coord1_10conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord1 done')

save_data_list += [ADMM_CUATRO_list10conv]

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
ALADIN_CUATRO_list10conv = []
s = 'ALADIN_CUATRO'
for i in range(1):
    System_dataAL10conv = ALADIN_Data(N, N_var, index_agents, global_ind)
    System_dataAL10conv.initialize(rho, N_it, z, list_fi, A_dict)
    System_dataAL10conv.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)
    ALADIN_CUATRO_list10conv += [System_dataAL10conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord2 done')

save_data_list += [ALADIN_CUATRO_list10conv]

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

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
y0 = np.array([f_BO(x0)])

pybobyqa10conv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

save_data_list += [pybobyqa10conv]
    
s = 'DIRECT'
DIRECT_List10conv = []
for i in range(N_runs): 
    DIRECT10conv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                    maxfun = N_it, constraints=1)
    for j in range(len(DIRECT10conv['f_best_so_far'])):
        if DIRECT10conv['f_best_so_far'][j] > float(y0):
            DIRECT10conv['f_best_so_far'][j] = float(y0)
    DIRECT_List10conv += [DIRECT10conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('DIRECT done')

save_data_list += [DIRECT_List10conv]

s = 'BO'
BO_List10conv = []
for i in range(N_runs):
    BO10conv = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,len(x0))), Y=y0.reshape((1,1)))
    BO10conv.run_optimization(max_iter=N_it, eps=0)
    BO_post10conv = preprocess_BO(BO10conv.Y.flatten(), y0, N_eval = N_it)
    BO_List10conv += [BO_post10conv]
    print(s + ' run ' + str(i+1) + ': Done')
    
save_data_list += [BO_List10conv]

dim = len(x0)
problem = 'MAC_conv_'

for k in range(len(s_list)):
    with open('../Data/' + problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'wb') as handle:
        pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 

data_dict = {}
for k in range(len(s_list)):
    with open('../Data/'+ problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'rb') as handle:
        data_dict[s_list[k]] = pickle.load(handle)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM'
out = postprocessing(ax1, ax2,  s, data_dict[s], pyo.value(res.obj), c='dodgerblue', N=N)
ax1, ax2 = out

s = 'ADMM_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='darkorange', N_it=N_it)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='red', N_it=N_it)
ax1, ax2 = out

s = 'ALADIN_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), ALADIN = True, c='darkviolet', N_it=N_it)
ax1, ax2 = out

s = 'GPyOpt'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), BO = True, c='saddlebrown', N_it=N_it)
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
ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

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
ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_conv_'
fig1.savefig('../Figures/' + problem + str(N_var) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")



N_var = 25

data_centr = {None: {'I': {None: list(np.arange(N_var)+1)}}}
res = centralized_nonconv(data_centr, dim=N_var)

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

save_data_list = []

ADMM_Scaled_system25nonconv = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system25nonconv.initialize_ADMM(rho/10, N_it, list_fi, z)
ADMM_Scaled_system25nonconv.solve_ADMM()

save_data_list += [postprocess_ADMM(ADMM_Scaled_system25nonconv)]

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

ADMM_CUATRO_list25nonconv = []
s = 'ADMM_CUATRO'
for i in range(1):
    Coordinator_ADMM_system25nonconv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
    Coordinator_ADMM_system25nonconv.initialize_Decomp(rho, N_it, list_fi, z)
    output_Coord1_25nonconv = Coordinator_ADMM_system25nonconv.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)  
    ADMM_CUATRO_list25nonconv += [output_Coord1_25nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord1 done')

save_data_list += [ADMM_CUATRO_list25nonconv]

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
ALADIN_CUATRO_list25nonconv = []
s = 'ALADIN_CUATRO'
for i in range(1):
    System_dataAL25nonconv = ALADIN_Data(N, N_var, index_agents, global_ind)
    System_dataAL25nonconv.initialize(rho, N_it, z, list_fi, A_dict)
    System_dataAL25nonconv.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)
    ALADIN_CUATRO_list25nonconv += [System_dataAL25nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord2 done')

save_data_list += [ALADIN_CUATRO_list25nonconv]

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

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
y0 = np.array([f_BO(x0)])

pybobyqa25nonconv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

save_data_list += [pybobyqa25nonconv]
    
s = 'DIRECT'
DIRECT_List25nonconv = []
for i in range(N_runs): 
    DIRECT25nonconv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                    maxfun = N_it, constraints=1)
    for j in range(len(DIRECT25nonconv['f_best_so_far'])):
        if DIRECT25nonconv['f_best_so_far'][j] > float(y0):
            DIRECT25nonconv['f_best_so_far'][j] = float(y0)
    DIRECT_List25nonconv += [DIRECT25nonconv]
    print(s + ' run ' + str(i+1) + ': Done')
print('DIRECT done')

save_data_list += [DIRECT_List25nonconv]

s = 'BO'
BO_List25nonconv = []
for i in range(N_runs):
    BO25nonconv = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,len(x0))), Y=y0.reshape((1,1)))
    BO25nonconv.run_optimization(max_iter=N_it, eps=0)
    BO_post25nonconv = preprocess_BO(BO25nonconv.Y.flatten(), y0, N_eval=N_it)
    BO_List25nonconv += [BO_post25nonconv]
    print(s + ' run ' + str(i+1) + ': Done')

save_data_list += [BO_List25nonconv]    

dim = len(x0)
problem = 'MAC_nonconv_'

for k in range(len(s_list)):
    with open('../Data/' + problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'wb') as handle:
        pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 


data_dict = {}
for k in range(len(s_list)):
    with open('../Data/'+ problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'rb') as handle:
        data_dict[s_list[k]] = pickle.load(handle)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM'
out = postprocessing(ax1, ax2,  s, data_dict[s], pyo.value(res.obj), c='dodgerblue', N=N)
ax1, ax2 = out

s = 'ADMM_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='darkorange', N_it=N_it)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='red', N_it=N_it)
ax1, ax2 = out

s = 'ALADIN_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), ALADIN = True, c='darkviolet', N_it=N_it)
ax1, ax2 = out

s = 'GPyOpt'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), BO = True, c='saddlebrown', N_it=N_it)
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
ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_nonconv_'
fig1.savefig('../Figures/' + problem + str(N_var) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")

N_var = 25

data_centr = {None: {'I': {None: list(np.arange(N_var)+1)}}}
res = centralized_conv(data_centr, dim=N_var)

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

save_data_list = []

ADMM_Scaled_system25conv = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system25conv.initialize_ADMM(rho/10, N_it, list_fi, z)
ADMM_Scaled_system25conv.solve_ADMM()

save_data_list += [postprocess_ADMM(ADMM_Scaled_system25conv)]

bounds = np.array([[0, 1]]*N_var)
x0 = np.zeros(N_var)+1/N_var
init_trust = 0.5 # 0.25
N_s = 40
beta = 0.95

ADMM_CUATRO_list25conv = []
s = 'ADMM_CUATRO'
for i in range(1):
    Coordinator_ADMM_system25conv = Coordinator_ADMM(N, N_var, index_agents, global_ind)
    Coordinator_ADMM_system25conv.initialize_Decomp(rho, N_it, list_fi, z)
    output_Coord1_25conv = Coordinator_ADMM_system25conv.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta, N_min_s = N_s)  
    ADMM_CUATRO_list25conv += [output_Coord1_25conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord1 done')

save_data_list += [ADMM_CUATRO_list25conv]

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
ALADIN_CUATRO_list25conv = []
s = 'ALADIN_CUATRO'
for i in range(1):
    System_dataAL25conv = ALADIN_Data(N, N_var, index_agents, global_ind)
    System_dataAL25conv.initialize(rho, N_it, z, list_fi, A_dict)
    System_dataAL25conv.solve(N_s, init_trust, mu = 1e7, beta_red = beta, bounds = bounds)
    ALADIN_CUATRO_list25conv += [System_dataAL25conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('Coord2 done')

save_data_list += [ALADIN_CUATRO_list25conv]

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

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (0., 1.)} for i in range(N_var)]
y0 = np.array([f_BO(x0)])

pybobyqa25conv = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

save_data_list += [pybobyqa25conv]
    
s = 'DIRECT'
DIRECT_List25conv = []
for i in range(N_runs): 
    DIRECT25conv =  DIRECTWrapper().solve(f_DIR, x0, bounds, \
                                    maxfun = N_it, constraints=1)
    for j in range(len(DIRECT25conv['f_best_so_far'])):
        if DIRECT25conv['f_best_so_far'][j] > float(y0):
            DIRECT25conv['f_best_so_far'][j] = float(y0)
    DIRECT_List25conv += [DIRECT25conv]
    print(s + ' run ' + str(i+1) + ': Done')
print('DIRECT done')

save_data_list += [DIRECT_List25conv]

s = 'BO'
BO_List25conv = []
for i in range(N_runs):
    BO25conv = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,len(x0))), Y=y0.reshape((1,1)))
    BO25conv.run_optimization(max_iter=N_it, eps=0)
    BO_post25conv = preprocess_BO(BO25conv.Y.flatten(), y0, N_eval=N_it)
    BO_List25conv += [BO_post25conv]
    print(s + ' run ' + str(i+1) + ': Done')
    
save_data_list += [BO_List25conv]

dim = len(x0)
problem = 'MAC_conv_'

for k in range(len(s_list)):
    with open('../Data/' + problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'wb') as handle:
        pickle.dump(save_data_list[k], handle, protocol=pickle.HIGHEST_PROTOCOL) 


data_dict = {}
for k in range(len(s_list)):
    with open('../Data/'+ problem + str(N_var) +'dim_'+ s_list[k] + '.pickle', 'rb') as handle:
        data_dict[s_list[k]] = pickle.load(handle)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM'
out = postprocessing(ax1, ax2,  s, data_dict[s], pyo.value(res.obj), c='dodgerblue', N=N)
ax1, ax2 = out

s = 'ADMM_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='darkorange', N_it=N_it)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), coord_input = True, c='red', N_it=N_it)
ax1, ax2 = out

s = 'ALADIN_CUATRO'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), ALADIN = True, c='darkviolet', N_it=N_it)
ax1, ax2 = out

s = 'GPyOpt'
out = postprocessing_List(ax1, ax2, s, data_dict[s], pyo.value(res.obj), BO = True, c='saddlebrown', N_it=N_it)
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
ax2.plot([1, N_it], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'MAC_conv_'
fig1.savefig('../Figures/' + problem + str(N_var) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N_var) + 'dim_evals.svg', format = "svg")







