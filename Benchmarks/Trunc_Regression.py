# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:33:43 2021

@author: dv516
"""

from Problems.Truncated_regression import f, regression
from Algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from Algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper
from Algorithms.ALADIN_Data import System as ALADIN_Data
from Algorithms.ADMM_Scaled_Consensus import System as ADMM_Scaled
# from Algorithms.Coordinator_Augmented import System as Coordinator_ADMM
from Algorithms.Coordinator_Augmented import System as Coordinator_ADMM
from Algorithms.CUATRO import CUATRO
from GPyOpt.methods import BayesianOptimization

import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo

from utilities import postprocessing, preprocess_BO, construct_A


dim = 2

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
rho_reg = 3

N = 2
xi = 1e-2
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 50

N_var = 2


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1 = Coordinator_ADMM_system.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)


A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL.solve(6, init_trust, mu = 1e7, infeas_start = True)


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

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_1'
out = postprocessing(ax1, ax2, s, output_Coord1, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA'
out = postprocessing(ax1, ax2, s, pybobyqa, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L'
out = postprocessing(ax1, ax2, s, DIRECT, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2'
out = postprocessing(ax1, ax2, s, System_dataAL, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO'
out = postprocessing(ax1, ax2, s, BO_post, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = 50
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")


dim = 4

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 100

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system4d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system4d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system4d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system4d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system4d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_4d = Coordinator_ADMM_system4d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)


A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL4d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL4d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL4d.solve(6, init_trust, mu = 1e7, infeas_start = True)


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

pybobyqa4d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT4d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post4d = preprocess_BO(BO.Y.flatten(), y0)


fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_4d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system4d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_1_4d'
out = postprocessing(ax1, ax2, s, output_Coord1_4d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_4d'
out = postprocessing(ax1, ax2, s, pybobyqa4d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_4d'
out = postprocessing(ax1, ax2, s, DIRECT4d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_4d'
out = postprocessing(ax1, ax2, s, System_dataAL4d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_4d'
out = postprocessing(ax1, ax2, s, BO_post4d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")



dim = 6

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
# xi = 1e-2
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 100

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system6d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system6d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system6d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system6d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system6d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_6d = Coordinator_ADMM_system6d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL6d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL6d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL6d.solve(6, init_trust, mu = 1e7, infeas_start = True)


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

pybobyqa6d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT6d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post6d = preprocess_BO(BO.Y.flatten(), y0)


fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_6d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system6d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_6d'
out = postprocessing(ax1, ax2, s, output_Coord1_6d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_6d'
out = postprocessing(ax1, ax2, s, pybobyqa6d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_6d'
out = postprocessing(ax1, ax2, s, DIRECT6d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_6d'
out = postprocessing(ax1, ax2, s, System_dataAL6d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_6d'
out = postprocessing(ax1, ax2, s, BO_post6d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")




dim = 8

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
xi = 1e-4
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 150

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system8d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system8d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system8d.solve_ADMM()


init_trust = 0.5
beta = 0.95

Coordinator_ADMM_system8d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system8d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_8d = Coordinator_ADMM_system8d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)


A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL8d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL8d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL8d.solve(6, init_trust, mu = 1e7, infeas_start = True)


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

pybobyqa8d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT8d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post8d = preprocess_BO(BO.Y.flatten(), y0)


fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_8d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system8d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_8d'
out = postprocessing(ax1, ax2, s, output_Coord1_8d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_8d'
out = postprocessing(ax1, ax2, s, pybobyqa8d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_8d'
out = postprocessing(ax1, ax2, s, DIRECT8d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_8d'
out = postprocessing(ax1, ax2, s, System_dataAL8d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_8d'
out = postprocessing(ax1, ax2, s, BO_post8d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()


problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")





# dim = 6

# N_data = 3000
# np.random.seed(0)
# x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
# H = np.random.normal(size=(N_data,dim))
# stdev = dim/10
# noise = np.random.normal(scale=stdev, size=(N_data,1))
# y = H @ x_ground+ noise
# # rho_reg = 3

# N = 2
# xi = 1e-2
# # xi=0.1

# data = {'H': H, 'y': y}

# data = {None: {
#                 'I': [i+1 for i in range(dim)],
#                 'N_data': [n+1 for n in range(N_data)],
#                 'y': {}, 'H': {}, 'xi': {None: xi},
#                 'N_all': {None: N_data}, 'N': {None: 2},
#                 'rho_reg': {None: rho_reg},
#              }
#        }

# for i in range(dim):
#     for n in range(N_data):
#         data[None]['H'][(n+1,i+1)] = float(H[n,i])
# for n in range(N_data): 
#     data[None]['y'][n+1] = float(y[n])

# res = regression(data)

# for i in res.I:
#     print(pyo.value(res.x[i]))
# print('Obj: ', pyo.value(res.obj))    

# rho = 10
# N_it = 100

# N_var = dim


# global_ind = list(np.arange(N_var)+1)
# index_agents = global_ind

# N_local = int(N_data/N)
# data_big = {ag+1: {} for ag in range(N)}
# for ag in range(N):
#     data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
#     data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
# x0 = np.array([0]*N_var)
# bounds = np.array([[-1, 1]]*N_var)
# z = {i: x0[i-1] for i in global_ind}

# index_agents = {i+1: global_ind for i in range(N)}

# def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

# list_fi = [f1, f2]

# ADMM_Scaled_system6d2ag = ADMM_Scaled(N, N_var, index_agents, global_ind)
# ADMM_Scaled_system6d2ag.initialize_ADMM(rho, N_it, list_fi, z)
# ADMM_Scaled_system6d2ag.solve_ADMM()


# init_trust = 0.5
# beta = 0.95

# Coordinator_ADMM_system6d2ag = Coordinator_ADMM(N, N_var, index_agents, global_ind)
# Coordinator_ADMM_system6d2ag.initialize_Decomp(rho, N_it, list_fi, z)
# output_Coord1_6d2ag = Coordinator_ADMM_system6d2ag.solve(CUATRO, x0, bounds, init_trust, 
#                             budget = N_it, beta_red = beta)


# A_dict = construct_A(index_agents, global_ind, N, only_global = True)
# System_dataAL6d2ag = ALADIN_Data(N, N_var, index_agents, global_ind)
# System_dataAL6d2ag.initialize(rho, N_it, z, list_fi, A_dict)
# System_dataAL6d2ag.solve(6, init_trust, mu = 1e7, infeas_start = True)


# def f_pbqa(x):
#     z_list = {i: [x[i-1]] for i in global_ind}
#     return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
# f_DIR = lambda x, grad: f_pbqa(x)
# def f_BO(x):
#     if x.ndim > 1:
#        x_temp = x[-1] 
#     else:
#        x_temp = x
#     # temp_dict = {i+1: x[:,i] for i in range(len(x))}
#     z_list = {i: [x_temp[i-1]] for i in global_ind}
#     return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

# pybobyqa6d2ag = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
#                                       maxfun=N_it, constraints=1, \
#                                       seek_global_minimum= True, \
#                                       objfun_has_noise=False)

# DIRECT6d2ag =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
#                                    constraints=1)

# domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

# y0 = np.array([f_BO(x0)])
# BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
# BO.run_optimization(max_iter=N_it)
# BO_post6d2ag = preprocess_BO(BO.Y.flatten(), y0)


# fig1 = plt.figure() 
# ax1 = fig1.add_subplot()  
# fig2 = plt.figure() 
# ax2 = fig2.add_subplot()  
# # ax2, fig2 = trust_fig(X, Y, Z, g)  

# s = 'ADMM_Scaled_6d2ag'
# out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system6d2ag, pyo.value(res.obj))
# ax1, ax2 = out

# s = 'CUATRO_6d2ag'
# out = postprocessing(ax1, ax2, s, output_Coord1_6d2ag, pyo.value(res.obj), coord_input = True)
# ax1, ax2 = out

# s = 'Py-BOBYQA_6d2ag'
# out = postprocessing(ax1, ax2, s, pybobyqa6d2ag, pyo.value(res.obj), coord_input = True)
# ax1, ax2 = out

# s = 'DIRECT-L_6d2ag'
# out = postprocessing(ax1, ax2, s, DIRECT6d2ag, pyo.value(res.obj), coord_input = True)
# ax1, ax2 = out

# s = 'CUATRO_2_6d2ag'
# out = postprocessing(ax1, ax2, s, System_dataAL6d2ag, pyo.value(res.obj), ALADIN = True)
# ax1, ax2 = out

# s = 'BO_6d2ag'
# out = postprocessing(ax1, ax2, s, BO_post6d2ag, pyo.value(res.obj), BO = True)
# ax1, ax2 = out

# N_it_temp = N_it
# # ax1.scatter
# # ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
# ax1.set_xlabel('Number of function evaluations')
# ax1.set_ylabel('Convergence')
# ax1.set_yscale('log')
# ax1.legend()

# # ax1.scatter
# # ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
# ax2.set_xlabel('Number of function evaluations')
# ax2.set_ylabel('Best function evaluation')
# ax2.set_yscale('log')
# ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
# ax2.legend()





dim = 6

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 4
xi = 1e-2
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: N},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 100

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 3, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 4, N_data, N, u_list = u_list, solver = solver, xi=xi)
list_fi = [f1, f2, f3, f4]

ADMM_Scaled_system6d4ag = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system6d4ag.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system6d4ag.solve_ADMM()

init_trust = 0.5
beta = 0.95

print('ADMM done')

Coordinator_ADMM_system6d4ag = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system6d4ag.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_6d4ag = Coordinator_ADMM_system6d4ag.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

print('Coord1 done')

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL6d4ag = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL6d4ag.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL6d4ag.solve(6, init_trust, mu = 1e7, infeas_start = True)

print('Coord2 done')

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

pybobyqa6d4ag = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

print('Py-BOBYQA done')
    
DIRECT6d4ag =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

print('DIRECT-L done')

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post6d4ag = preprocess_BO(BO.Y.flatten(), y0)

print('BO done')

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_6d4ag'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system6d4ag, pyo.value(res.obj), N=N, c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_6d4ag'
out = postprocessing(ax1, ax2, s, output_Coord1_6d4ag, pyo.value(res.obj), coord_input = True, N=N, c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA_6d4ag'
out = postprocessing(ax1, ax2, s, pybobyqa6d4ag, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L_6d4ag'
out = postprocessing(ax1, ax2, s, DIRECT6d4ag, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2_6d4ag'
out = postprocessing(ax1, ax2, s, System_dataAL6d4ag, pyo.value(res.obj), ALADIN = True, init=float(y0), N=N, c='darkviolet')
ax1, ax2 = out

s = 'BO_6d4ag'
out = postprocessing(ax1, ax2, s, BO_post6d4ag, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()


problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")




dim = 6

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 8
xi = 1e-2
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: N},
                'rho_reg': {None: rho_reg},
             }
       }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 100

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f3(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 3, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f4(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 4, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f5(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 5, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f6(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 6, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f7(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 7, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f8(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 8, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2, f3, f4, f5, f6, f7, f8]

ADMM_Scaled_system6d8ag = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system6d8ag.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system6d8ag.solve_ADMM()


init_trust = 0.5
beta = 0.95

Coordinator_ADMM_system6d8ag = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system6d8ag.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_6d8ag = Coordinator_ADMM_system6d8ag.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)


A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL6d8ag = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL6d8ag.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL6d8ag.solve(6, init_trust, mu = 1e7, infeas_start=True)


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

pybobyqa6d8ag = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

DIRECT6d8ag =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post6d8ag = preprocess_BO(BO.Y.flatten(), y0)

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_6d8ag'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system6d8ag, pyo.value(res.obj), N=N, c='dodgerblue')
ax1, ax2 = out

s = 'CUATRO_6d8ag'
out = postprocessing(ax1, ax2, s, output_Coord1_6d8ag, pyo.value(res.obj), coord_input = True, N=N,  c='darkorange')
ax1, ax2 = out

s = 'Py-BOBYQA_6d8ag'
out = postprocessing(ax1, ax2, s, pybobyqa6d8ag, pyo.value(res.obj), coord_input = True, c='green')
ax1, ax2 = out

s = 'DIRECT-L_6d8ag'
out = postprocessing(ax1, ax2, s, DIRECT6d8ag, pyo.value(res.obj), coord_input = True, c='red')
ax1, ax2 = out

s = 'CUATRO_2_6d8ag'
out = postprocessing(ax1, ax2, s, System_dataAL6d8ag, pyo.value(res.obj), ALADIN = True, init=float(y0), N=N, c='darkviolet')
ax1, ax2 = out

s = 'BO_6d8ag'
out = postprocessing(ax1, ax2, s, BO_post6d8ag, pyo.value(res.obj), BO = True, c='saddlebrown')
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()


problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")






# dim = 6

# N_data = 3000
# np.random.seed(0)
# x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
# H = np.random.normal(size=(N_data,dim))
# stdev = dim/10
# noise = np.random.normal(scale=stdev, size=(N_data,1))
# y = H @ x_ground+ noise
# # rho_reg = 3

# N = 16
# xi = 1e-2
# # xi=0.1

# data = {'H': H, 'y': y}

# data = {None: {
#                 'I': [i+1 for i in range(dim)],
#                 'N_data': [n+1 for n in range(N_data)],
#                 'y': {}, 'H': {}, 'xi': {None: xi},
#                 'N_all': {None: N_data}, 'N': {None: N},
#                 'rho_reg': {None: rho_reg},
#              }
#        }

# for i in range(dim):
#     for n in range(N_data):
#         data[None]['H'][(n+1,i+1)] = float(H[n,i])
# for n in range(N_data): 
#     data[None]['y'][n+1] = float(y[n])

# res = regression(data)

# for i in res.I:
#     print(pyo.value(res.x[i]))
# print('Obj: ', pyo.value(res.obj))    

# rho = 10
# N_it = 100

# N_var = dim


# global_ind = list(np.arange(N_var)+1)
# index_agents = global_ind

# N_local = int(N_data/N)
# data_big = {ag+1: {} for ag in range(N)}
# for ag in range(N):
#     data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
#     data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
# x0 = np.array([0]*N_var)
# bounds = np.array([[-1, 1]]*N_var)
# z = {i: x0[i-1] for i in global_ind}

# index_agents = {i+1: global_ind for i in range(N)}

# def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f3(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 3, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f4(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 4, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f5(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 5, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f6(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 6, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f7(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 7, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f8(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 8, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f9(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 9, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f10(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 10, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f11(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 11, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f12(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 12, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f13(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 13, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f14(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 14, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f15(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 15, N_data, N, u_list = u_list, solver = solver, xi=xi)
# def f16(z_list, rho, global_ind, index, u_list = None, solver = False):
#     return f(z_list, rho, global_ind, index, data_big, 16, N_data, N, u_list = u_list, solver = solver, xi=xi)

# list_fi = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16]

# ADMM_Scaled_system6d16ag = ADMM_Scaled(N, N_var, index_agents, global_ind)
# ADMM_Scaled_system6d16ag.initialize_ADMM(rho, N_it, list_fi, z)
# ADMM_Scaled_system6d16ag.solve_ADMM()


# init_trust = 0.5
# beta = 0.95

# Coordinator_ADMM_system6d16ag = Coordinator_ADMM(N, N_var, index_agents, global_ind)
# Coordinator_ADMM_system6d16ag.initialize_Decomp(rho, N_it, list_fi, z)
# output_Coord1_6d16ag = Coordinator_ADMM_system6d16ag.solve(CUATRO, x0, bounds, init_trust, 
#                             budget = N_it, beta_red = beta)


# A_dict = construct_A(index_agents, global_ind, N, only_global = True)
# System_dataAL6d16ag = ALADIN_Data(N, N_var, index_agents, global_ind)
# System_dataAL6d16ag.initialize(rho, N_it, z, list_fi, A_dict)
# System_dataAL6d16ag.solve(6, init_trust, mu = 1e7, infeas_start = True)


# def f_pbqa(x):
#     z_list = {i: [x[i-1]] for i in global_ind}
#     return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi]), [0]
# f_DIR = lambda x, grad: f_pbqa(x)
# def f_BO(x):
#     if x.ndim > 1:
#        x_temp = x[-1] 
#     else:
#        x_temp = x
#     # temp_dict = {i+1: x[:,i] for i in range(len(x))}
#     z_list = {i: [x_temp[i-1]] for i in global_ind}
#     return np.sum([pyo.value(f(z_list, rho, global_ind, global_ind).obj) for f in list_fi])

# pybobyqa6d16ag = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
#                                       maxfun=N_it, constraints=1, \
#                                       seek_global_minimum= True, \
#                                       objfun_has_noise=False)

# DIRECT6d16ag =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
#                                    constraints=1)

# domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

# y0 = np.array([f_BO(x0)])
# BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
# BO.run_optimization(max_iter=N_it)
# BO_post6d16ag = preprocess_BO(BO.Y.flatten(), y0)


# fig1 = plt.figure() 
# ax1 = fig1.add_subplot()  
# fig2 = plt.figure() 
# ax2 = fig2.add_subplot()  
# # ax2, fig2 = trust_fig(X, Y, Z, g)  

# s = 'ADMM_Scaled_6d16ag'
# out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system6d16ag, pyo.value(res.obj), N=N)
# ax1, ax2 = out

# s = 'CUATRO_6d16ag'
# out = postprocessing(ax1, ax2, s, output_Coord1_6d16ag, pyo.value(res.obj), coord_input = True, N=N)
# ax1, ax2 = out

# s = 'Py-BOBYQA_6d16ag'
# out = postprocessing(ax1, ax2, s, pybobyqa6d16ag, pyo.value(res.obj), coord_input = True)
# ax1, ax2 = out

# s = 'DIRECT-L_6d16ag'
# out = postprocessing(ax1, ax2, s, DIRECT6d16ag, pyo.value(res.obj), coord_input = True)
# ax1, ax2 = out

# s = 'CUATRO_2_6d16ag'
# out = postprocessing(ax1, ax2, s, System_dataAL6d16ag, pyo.value(res.obj), ALADIN = True)
# ax1, ax2 = out

# s = 'BO_6d16ag'
# out = postprocessing(ax1, ax2, s, BO_post6d16ag, pyo.value(res.obj), BO = True)
# ax1, ax2 = out

# N_it_temp = N_it
# # ax1.scatter
# # ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
# ax1.set_xlabel('Number of function evaluations')
# ax1.set_ylabel('Convergence')
# ax1.set_yscale('log')
# ax1.legend()

# # ax1.scatter
# # ax2.plot(np.array([1, 51]), np.array([actual_f, actual_f]), c = 'red', label = 'optimum')
# ax2.set_xlabel('Number of function evaluations')
# ax2.set_ylabel('Best function evaluation')
# ax2.set_yscale('log')
# ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
# ax2.legend()

# problem = 'Truncated_Regression_'
# fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
# fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")




dim = 10

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
xi = 0
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
              }
        }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 200

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system10d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system10d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system10d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system10d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system10d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_10d = Coordinator_ADMM_system10d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

print('Coord1 done')

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL10d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL10d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL10d.solve(6, init_trust, mu = 1e7, infeas_start = True)

print('Coord2 done')

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

pybobyqa10d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

print('Py-BOBYQA done')
    
DIRECT10d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                   constraints=1)

print('DIRECT done')

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post10d = preprocess_BO(BO.Y.flatten(), y0)

print('BO done')

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_10d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system10d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_10d'
out = postprocessing(ax1, ax2, s, output_Coord1_10d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_10d'
out = postprocessing(ax1, ax2, s, pybobyqa10d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_10d'
out = postprocessing(ax1, ax2, s, DIRECT10d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_10d'
out = postprocessing(ax1, ax2, s, System_dataAL10d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_10d'
out = postprocessing(ax1, ax2, s, BO_post10d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()


problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")




## Run till here





dim = 15

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
xi = 0
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
              }
        }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 200

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system15d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system15d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system15d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system15d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system15d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_15d = Coordinator_ADMM_system15d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

print('Coord1 done')

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL15d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL15d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL15d.solve(6, init_trust, mu = 1e7, infeas_start = True)

print('Coord2 done')

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

pybobyqa15d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

print('Py-BOBYQA done')
    
DIRECT15d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                    constraints=1)

print('DIRECT done')

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post15d = preprocess_BO(BO.Y.flatten(), y0)

print('BO done')

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_15d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system15d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_15d'
out = postprocessing(ax1, ax2, s, output_Coord1_15d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_15d'
out = postprocessing(ax1, ax2, s, pybobyqa15d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_15d'
out = postprocessing(ax1, ax2, s, DIRECT15d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_15d'
out = postprocessing(ax1, ax2, s, System_dataAL15d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_15d'
out = postprocessing(ax1, ax2, s, BO_post15d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")



dim = 20

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
xi = 0
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
              }
        }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 200

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system20d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system20d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system20d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system20d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system20d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_20d = Coordinator_ADMM_system20d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

print('Coord1 done')

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL20d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL20d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL20d.solve(6, init_trust, mu = 1e7, infeas_start = True)

print('Coord2 done')

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

pybobyqa20d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

print('Py-BOBYQA done')
    
DIRECT20d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                    constraints=1)

print('DIRECT done')

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post20d = preprocess_BO(BO.Y.flatten(), y0)

print('BO done')

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_20d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system20d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_20d'
out = postprocessing(ax1, ax2, s, output_Coord1_20d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_20d'
out = postprocessing(ax1, ax2, s, pybobyqa20d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_20d'
out = postprocessing(ax1, ax2, s, DIRECT20d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_20d'
out = postprocessing(ax1, ax2, s, System_dataAL20d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_20d'
out = postprocessing(ax1, ax2, s, BO_post20d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")




dim = 50

N_data = 3000
np.random.seed(0)
x_ground = np.random.uniform(low=-1, high=1, size=(dim,1))
H = np.random.normal(size=(N_data,dim))
stdev = dim/10
noise = np.random.normal(scale=stdev, size=(N_data,1))
y = H @ x_ground+ noise
# rho_reg = 3

N = 2
xi = 0
# xi=0.1

data = {'H': H, 'y': y}

data = {None: {
                'I': [i+1 for i in range(dim)],
                'N_data': [n+1 for n in range(N_data)],
                'y': {}, 'H': {}, 'xi': {None: xi},
                'N_all': {None: N_data}, 'N': {None: 2},
                'rho_reg': {None: rho_reg},
              }
        }

for i in range(dim):
    for n in range(N_data):
        data[None]['H'][(n+1,i+1)] = float(H[n,i])
for n in range(N_data): 
    data[None]['y'][n+1] = float(y[n])

res = regression(data)

for i in res.I:
    print(pyo.value(res.x[i]))
print('Obj: ', pyo.value(res.obj))    

rho = 10
N_it = 200

N_var = dim


global_ind = list(np.arange(N_var)+1)
index_agents = global_ind

N_local = int(N_data/N)
data_big = {ag+1: {} for ag in range(N)}
for ag in range(N):
    data_big[ag+1]['y'] = y[ag*N_local:(ag+1)*N_local]
    data_big[ag+1]['H'] = H[ag*N_local:(ag+1)*N_local]
 
x0 = np.array([0]*N_var)
bounds = np.array([[-1, 1]]*N_var)
z = {i: x0[i-1] for i in global_ind}

index_agents = {i+1: global_ind for i in range(N)}

def f1(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 1, N_data, N, u_list = u_list, solver = solver, xi=xi)
def f2(z_list, rho, global_ind, index, u_list = None, solver = False):
    return f(z_list, rho, global_ind, index, data_big, 2, N_data, N, u_list = u_list, solver = solver, xi=xi)

list_fi = [f1, f2]

ADMM_Scaled_system50d = ADMM_Scaled(N, N_var, index_agents, global_ind)
ADMM_Scaled_system50d.initialize_ADMM(rho, N_it, list_fi, z)
ADMM_Scaled_system50d.solve_ADMM()


init_trust = 0.5
beta = 0.9

Coordinator_ADMM_system50d = Coordinator_ADMM(N, N_var, index_agents, global_ind)
Coordinator_ADMM_system50d.initialize_Decomp(rho, N_it, list_fi, z)
output_Coord1_50d = Coordinator_ADMM_system50d.solve(CUATRO, x0, bounds, init_trust, 
                            budget = N_it, beta_red = beta)

print('Coord1 done')

A_dict = construct_A(index_agents, global_ind, N, only_global = True)
System_dataAL50d = ALADIN_Data(N, N_var, index_agents, global_ind)
System_dataAL50d.initialize(rho, N_it, z, list_fi, A_dict)
System_dataAL50d.solve(6, init_trust, mu = 1e7, infeas_start = True)

print('Coord2 done')

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

pybobyqa50d = PyBobyqaWrapper().solve(f_pbqa, x0, bounds=bounds.T, \
                                      maxfun=N_it, constraints=1, \
                                      seek_global_minimum= True, \
                                      objfun_has_noise=False)

print('Py-BOBYQA done')
    
DIRECT50d =  DIRECTWrapper().solve(f_DIR, x0, bounds, maxfun = N_it, 
                                    constraints=1)

print('DIRECT done')

domain = [{'name': 'var_'+str(i+1), 'type': 'continuous', 'domain': (-1,1)} for i in range(N_var)]

y0 = np.array([f_BO(x0)])
BO = BayesianOptimization(f=f_BO, domain=domain, X=x0.reshape((1,dim)), Y=y0.reshape((1,1)))
BO.run_optimization(max_iter=N_it)
BO_post50d = preprocess_BO(BO.Y.flatten(), y0)

print('BO done')

fig1 = plt.figure() 
ax1 = fig1.add_subplot()  
fig2 = plt.figure() 
ax2 = fig2.add_subplot()  
# ax2, fig2 = trust_fig(X, Y, Z, g)  

s = 'ADMM_Scaled_50d'
out = postprocessing(ax1, ax2,  s, ADMM_Scaled_system50d, pyo.value(res.obj))
ax1, ax2 = out

s = 'CUATRO_50d'
out = postprocessing(ax1, ax2, s, output_Coord1_50d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'Py-BOBYQA_50d'
out = postprocessing(ax1, ax2, s, pybobyqa50d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'DIRECT-L_50d'
out = postprocessing(ax1, ax2, s, DIRECT50d, pyo.value(res.obj), coord_input = True)
ax1, ax2 = out

s = 'CUATRO_2_50d'
out = postprocessing(ax1, ax2, s, System_dataAL50d, pyo.value(res.obj), ALADIN = True)
ax1, ax2 = out

s = 'BO_50d'
out = postprocessing(ax1, ax2, s, BO_post50d, pyo.value(res.obj), BO = True)
ax1, ax2 = out

N_it_temp = N_it
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
ax2.plot([1, N_it_temp], [pyo.value(res.obj), pyo.value(res.obj)], '--k', label = 'Centralized')
ax2.legend()

problem = 'Truncated_Regression_'
fig1.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_conv.svg', format = "svg")
fig2.savefig('../Figures/' + problem + str(N) + 'ag_' + str(dim) + 'dim_evals.svg', format = "svg")






