# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:52:04 2021

@author: dv516
"""

import pyomo.environ as pyo
import numpy as np
# import matplotlib.pyplot as plt

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


# def f1(z_list, u_list, rho, global_ind, index):
    
#     data = {None: {
#                   'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
#                   'rho': {None: rho}, 'I': {None: index}
#                 }
#           }
    
#     for idx in global_ind:
#         data[None]['z'][idx] = z_list[idx][-1]
#         data[None]['u'][idx] = u_list[idx][-1]
    
#     return SP1(data)

# def f2(z_list, u_list, rho, global_ind, index):

#     data = {None: {
#                   'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
#                   'rho': {None: rho}, 'I': {None: index}
#                 }
#           }
    
#     for idx in global_ind:
#         data[None]['z'][idx] = z_list[idx][-1]
#         data[None]['u'][idx] = u_list[idx][-1]

#     return SP2(data)


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
    
    
    def initialize_ADMM(self, rho, N_it, fi_list, z, u = 0): 
        self.rho = rho
        self.N_it = N_it
        self.f_list = fi_list
        self.prim_r = []
        self.dual_r = []
        
        self.systems = {}
        self.u_list = {}
        self.obj = {}
        
        if len(z) != len(self.global_ind):
            raise ValueError('z should have as many elements as global_ind')
        
        for i in range(self.N):
            self.systems[i+1] = {} ; self.u_list[i+1] = {}
            self.obj[i+1] = []
            for j in range(self.N_var):
                if j+1 in self.idx_agents[i+1]:
                    self.systems[i+1][j+1] = []
                if j+1 in self.global_ind:
                    self.u_list[i+1][j+1] = [u]
        
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
    
    
    def update_lists(self):
        for idx in self.global_ind:
            self.z_list[idx] += [np.mean(self.z_temp[idx])]
            for i in range(self.N):
                self.u_list[i+1][idx] += [self.u_list[i+1][idx][-1] + self.z_temp[idx][i] - \
                                          np.mean(self.z_temp[idx])]
    
    def solve_subproblems(self):
        for i in range(self.N):
            #print(z_list, u_list, rho)
            instance = self.f_list[i](self.z_list, self.rho, self.global_ind, 
                                      self.idx_agents[i+1], u_list = self.u_list[i+1])
            self.obj[i+1] += [pyo.value(instance.obj)]
            for j in instance.x:
                self.systems[i+1][j] += [pyo.value(instance.x[j])]
    
    def solve_ADMM(self):
        for k in range(self.N_it):
            self.solve_subproblems()    
            self.compute_residuals()
            self.update_lists()
        



# #u = 0
# rho = 50
# N_it = 50

# N = 2
# N_var = 3
# list_fi = [f1, f2]

# global_ind = [3]
# index_agents = {1: [1, 3], 2: [2, 3]}
# z = {3: 6}

# test_system = System(N, N_var, index_agents, global_ind)
# test_system.initialize_ADMM(rho, N_it, list_fi, z)
# test_system.solve_ADMM()

# systems = test_system.systems
# z_list = test_system.z_list
# prim_r = test_system.prim_r
# dual_r = test_system.dual_r

# actual_f = 13.864179350870021
# actual_x = 0.398
# # x_list = np.array(z_list[3])

# obj_arr = np.sum(np.array([test_system.obj[i+1] for i in range(N)]), axis = 0)
# z_arr = np.array(test_system.z_list[global_ind[0]])
# conv_arr = (obj_arr - actual_f)**2 

# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.step(np.arange(len(obj_arr))+1, conv_arr, where = 'post')
# ax.set_yscale('log')
# # ax.legend()

# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.scatter(z_arr[1:], conv_arr)    
# ax.scatter(actual_x, actual_f, c = 'red') 
# ax.set_yscale('log')
# ax.legend()

    
# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.plot(np.arange(N_it)+1, np.array(systems[1][1]), label = 'S1: x1')
# ax.plot(np.arange(N_it)+1, np.array(systems[1][3]), label = 'S1: x3')
# ax.plot(np.arange(N_it)+1, np.array(systems[2][2]), label = 'S2: x2')
# ax.plot(np.arange(N_it)+1, np.array(systems[2][3]), label = 'S2: x3')
# ax.plot(np.arange(N_it)+1, np.array(z_list[3])[1:], label = 'z')
# ax.legend()


# fig = plt.figure() 
# ax = fig.add_subplot() 
# ax.plot(np.arange(N_it)+1, np.array(prim_r), label = 'primal residual')
# ax.plot(np.arange(N_it)+1, np.array(dual_r), label = 'dual residual')
# ax.set_yscale('log')
# ax.legend()

    
# print('Final variables: ')
# print('x1: ', systems[1][1][-1])
# print('x2: ', systems[2][2][-1])
# print('z or x3: ', z_list[3][-1])
    
    
    
    
    
    
    
    




