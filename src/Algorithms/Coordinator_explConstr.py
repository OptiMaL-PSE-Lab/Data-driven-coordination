# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 23:08:58 2021

@author: dv516
"""

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
# import matplotlib.pyplot as plt

# from CUATRO import CUATRO



# def centralised(data):
#     model = pyo.AbstractModel()
#     model.i = pyo.RangeSet(1, 3)
#     model.x_init = pyo.Param(model.i)
#     model.x = pyo.Var(model.i, within = pyo.Reals, bounds = (-10, 10))

    
#     def o(m):
#         return (m.x[1] - 7)**2 + (m.x[1]*m.x[3]-3)**2 + (m.x[2]+2)**2 + \
#                 (m.x[2]*m.x[3]-2)**2
#     model.obj = pyo.Objective(rule = o)
    
#     ins = model.create_instance(data)
    
#     for j in ins.i:
#         ins.x[j] = ins.x_init[j]
    
#     #solver = pyo.SolverFactory('cbc')
#     solver = pyo.SolverFactory('ipopt')
#     solver.solve(ins)
    
#     return ins


# def SP1(data, return_solver = False):
#     model = pyo.AbstractModel()
#     #model.x_init = pyo.Param()
#     model.z_I = pyo.Set()
#     model.z = pyo.Param(model.z_I)
#     #model.u = pyo.Param(model.z_I)
#     model.rho = pyo.Param()
#     model.I = pyo.Set()
#     model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

#     if return_solver:
#         def o(m):
#             return (m.x[1] - 7)**2 + (m.x[1]*m.z[3]-3)**2 + \
#                 m.rho/2*sum((m.x[j] - m.z[j] )**2 for j in m.z_I)
#         model.obj = pyo.Objective(rule = o)
#     else:
#         def o(m):
#             return (m.x[1] - 7)**2 + (m.x[1]*m.x[3]-3)**2 + \
#                 m.rho/2*sum((m.x[j] - m.z[j] )**2 for j in m.z_I)
#         model.obj = pyo.Objective(rule = o)
    
#     def g(m):
#         return -m.x[1] <= 0
#     model.ineq1 = pyo.Constraint(rule = g)

#     if return_solver:
#         def h(m):
#             return m.x[1] + m.z[3] == 5
#         model.eq1 = pyo.Constraint(rule = h)
#     else:
#         def h(m):
#             return m.x[1] + m.x[3] == 5
#         model.eq1 = pyo.Constraint(rule = h)

#     ins = model.create_instance(data)
    
#     #ins.x = ins.x_init
    
#     #solver = pyo.SolverFactory('cbc')
#     solver = pyo.SolverFactory('ipopt')
#     res = solver.solve(ins)
    
#     if return_solver:
#         return ins, res
#     else:
#         return ins

# def SP2(data, return_solver = False):
#     model = pyo.AbstractModel()
#     #model.x_init = pyo.Param()
#     model.z_I = pyo.Set()
#     model.z = pyo.Param(model.z_I)
#     #model.u = pyo.Param(model.z_I)
#     model.rho = pyo.Param()
#     model.I = pyo.Set()
#     model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

#     if return_solver:
#         def o(m):
#             return (m.x[2] + 2)**2 + (m.x[2]*m.z[3]-2)**2 + \
#                 m.rho/2*sum((m.x[3] - m.z[j] )**2 for j in m.z_I)
#         model.obj = pyo.Objective(rule = o)
#     else:
#         def o(m):
#             return (m.x[2] + 2)**2 + (m.x[2]*m.x[3]-2)**2 + \
#                 m.rho/2*sum((m.x[3] - m.z[j] )**2 for j in m.z_I)
#         model.obj = pyo.Objective(rule = o)

#     ins = model.create_instance(data)
    
#     #ins.x = ins.x_init
    
#     #solver = pyo.SolverFactory('cbc')
#     solver = pyo.SolverFactory('ipopt')
#     res = solver.solve(ins)
    
#     if return_solver:
#         return ins, res
#     else:
#         return ins


# def f1(z_list, rho, global_ind, index, solver):
    
#     data = {None: {
#                   'z': {}, 'z_I': {None: global_ind}, 
#                   'rho': {None: rho}, 'I': {None: index}
#                 }
#           }
    
#     for idx in global_ind:
#         data[None]['z'][idx] = z_list[idx][-1]
#         #data[None]['u'][idx] = u_list[idx][-1]
    
#     return SP1(data, return_solver = solver)

# def f2(z_list, rho, global_ind, index, solver):

#     data = {None: {
#                   'z': {}, 'z_I': {None: global_ind}, 
#                   'rho': {None: rho}, 'I': {None: index}
#                 }
#           }
    
#     for idx in global_ind:
#         data[None]['z'][idx] = z_list[idx][-1]
#         #data[None]['u'][idx] = u_list[idx][-1]

#     return SP2(data, return_solver = solver)


class System:
    def __init__(self, N_agents, N_var, index_agents, global_ind):
        """
        :N_agents: Number of subsystems
        :index_agents: Dictionary where index_agents[1] gives the list of 
                      agent 1 variable indices as e.g. [1, 3]
        :global_ind:   list of indices for global variables e.g. [3]
        """
        
        self.N = N_agents
        
        if len(index_agents) != N_agents:
            raise ValueError('index_agents list should have as many items as N_agents')
        
        self.idx_agents = index_agents
        self.global_ind = global_ind
        self.N_var = N_var
    
    
    def initialize_Decomp(self, rho, N_it, fi_list, z, rho_inc = 1): 
        self.rho = rho
        self.N_it = N_it
        self.f_list = fi_list
        self.prim_r = []
        self.dual_r = []
        
        self.rho_inc = rho_inc
        
        self.systems = {}
        #self.u_list = {}
        
        if len(z) != len(self.global_ind):
            raise ValueError('z should have as many elements as global_ind')
        
        for i in range(self.N):
            self.systems[i+1] = {} #; self.u_list[i+1] = {}
            for j in range(self.N_var):
                if j+1 in self.idx_agents[i+1]:
                    self.systems[i+1][j+1] = []
                # if j+1 in self.global_ind:
                #     self.u_list[i+1][j+1] = [u]
        
        self.z_list = {} ; self.z_temp = {}
        for i in self.global_ind:
            self.z_list[i] = [z[i]] ; self.z_temp[i] = [] 
        
    
    def compute_residuals(self):
        for idx in self.global_ind:
            self.z_temp[idx] = [self.systems[i+1][idx][-1] for i in range(self.N)]
        self.prim_r += [ np.linalg.norm([np.linalg.norm([self.systems[i+1][idx][-1] - \
                         np.mean(self.z_temp[idx]) for i in range(self.N)]) for idx in self.global_ind])]
        self.dual_r += [ self.rho*np.linalg.norm([np.linalg.norm(np.mean(self.z_temp[idx]) - \
                         self.z_list[idx][-1]) for idx in self.global_ind])]
        
    
    def solve_subproblems(self):
        self.obj = 0
        self.conv = 0
        for i in range(self.N):
            #print(z_list, u_list, rho)
            try: 
                instance, s = self.f_list[i](self.z_list, self.rho, 
                                          self.global_ind, self.idx_agents[i+1], solver = True)
            
                if (s.solver.status != SolverStatus.ok) or (s.solver.termination_condition != TerminationCondition.optimal):
                    self.conv = 1
            except:
                instance = self.f_list[i](self.z_list, self.rho, 
                                          self.global_ind, self.idx_agents[i+1], solver = True)
            
            for j in instance.x:
                self.systems[i+1][j] += [pyo.value(instance.x[j])]

            self.obj += pyo.value(instance.obj)
    
    def f_surr(self, x):
        
        for i in range(len(self.global_ind)):
            self.z_list[self.global_ind[i]] += [x[i]]
            
        self.solve_subproblems()    
        self.compute_residuals()
        self.rho *= self.rho_inc
        
        return self.obj, [self.conv]
        
    def solve(self, solver, x0, bounds, init_trust, budget = 100, 
              N_min_s = 6, beta_red = 0.9, rnd_seed = 0, 
              method = 'local', constr = 'Discrimination',
              ineq_known = None, eq_known = None):
        self.obj_list = []
        
        dict_out = solver(self.f_surr, x0, init_trust, bounds = bounds, max_f_eval = budget, \
                           N_min_samples = N_min_s, beta_red = beta_red, \
                           rnd = rnd_seed, method = method, \
                           constr_handling = constr, 
                           eq_known = eq_known,
                           ineq_known = ineq_known)
        
        return dict_out


# #u = 0
# rho = 500
# N_it = 50

# N = 2
# N_var = 3
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
# z = {3: 4.5}

# x0 = np.array([z[3]])
# bounds = np.array([[-10, 10]])
# init_trust = 1

# test_system = System(N, N_var, index_agents, global_ind)
# test_system.initialize_Decomp(rho, N_it, list_fi, z)
# output = test_system.solve(CUATRO, x0, bounds, init_trust, 
#                            budget = N_it, beta_red = 0.5)    

# f = np.array(output['f_store'])
# x_list = np.array([output['x_store']])
# f_best = np.array(output['f_best_so_far'])
# ind_best = np.array(output['samples_at_iteration'])


# #test_system.solve_Decomp()

# # systems = test_system.systems
# # z_list = test_system.z_list
# # prim_r = test_system.prim_r
# # dual_r = test_system.dual_r

# actual_f = 13.864179350870021
# actual_x = 0.398
# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.step(ind_best, (f_best - actual_f)**2, where = 'post')
# ax.scatter(actual_x, 0, c = 'red')
# ax.set_yscale('log')


# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.scatter(x_list, f)
# ax.scatter(actual_x, actual_f, c = 'red')
# ax.set_yscale('log')




# # ax.plot(np.arange(N_it)+1, np.array(systems[1][1]), label = 'S1: x1')
# # ax.plot(np.arange(N_it)+1, np.array(systems[1][3]), label = 'S1: x3')
# # ax.plot(np.arange(N_it)+1, np.array(systems[2][2]), label = 'S2: x2')
# # ax.plot(np.arange(N_it)+1, np.array(systems[2][3]), label = 'S2: x3')
# # ax.plot(np.arange(N_it)+1, np.array(z_list[3])[1:], label = 'z')
# # ax.legend()


# # fig = plt.figure() 
# # ax = fig.add_subplot() 
# # ax.plot(np.arange(N_it)+1, np.array(prim_r), label = 'primal residual')
# # ax.plot(np.arange(N_it)+1, np.array(dual_r), label = 'dual residual')
# # ax.set_yscale('log')
# # ax.legend()
    
# # print('Final variables: ')
# # print('x1: ', systems[1][1][-1])
# # print('x2: ', systems[2][2][-1])
# # print('z or x3: ', z_list[3][-1])
    
    