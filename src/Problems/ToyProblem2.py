# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:27:06 2021

@author: dv516
"""

import pyomo.environ as pyo


def centralised_P2(data):
    model = pyo.AbstractModel()
    model.i = pyo.RangeSet(1, 10)
    #model.x_init = pyo.Param(model.i)
    model.x = pyo.Var(model.i, within = pyo.Reals, bounds = (-10, 10))

    def g1(m):
        return -105 + 4*m.x[1] + 5*m.x[2] - 3*m.x[7] + 9*m.x[8] <= 0
    model.ineq1 = pyo.Constraint(rule = g1)

    def g2(m):
        return 10*m.x[1] - 8*m.x[2] - 17*m.x[7] + 2*m.x[8] <= 0
    model.ineq2 = pyo.Constraint(rule = g2)
    
    def g3(m):
        return -8*m.x[1] + 2*m.x[2] + 5*m.x[9] - 2*m.x[10] - 12 <= 0
    model.ineq3 = pyo.Constraint(rule = g3)
    
    def g4(m):
        return 3*(m.x[1] - 2)**2 + 4*(m.x[2] - 3)**2 + 2*m.x[3]**2 - 7*m.x[4] - 120 <= 0
    model.ineq4 = pyo.Constraint(rule = g4)

    def g5(m):
        return 5*m.x[1]**2 + 8*m.x[2] + (m.x[3] - 6)**2 - 2*m.x[4] - 40 <= 0
    model.ineq5 = pyo.Constraint(rule = g5)

    def g6(m):
        return m.x[1]**2 + 2*(m.x[2] - 2)**2 - 2*m.x[1]*m.x[2] + 14*m.x[5] - 6*m.x[6] <= 0
    model.ineq6 = pyo.Constraint(rule = g6)
    
    def g7(m):
        return 0.5*(m.x[1] - 8)**2 + 2*(m.x[2] - 4)**2 + 3*m.x[5]**2 - m.x[6] - 30 <= 0
    model.ineq7 = pyo.Constraint(rule = g7)
    
    def g8(m):
        return -3*m.x[1] + 6*m.x[2] + 12*(m.x[9] - 8)**2 - 7*m.x[10] <= 0
    model.ineq8 = pyo.Constraint(rule = g8)

    def o(m):
        return m.x[1]**2 + m.x[2]**2 + m.x[1]*m.x[2] -14*m.x[1] - 16*m.x[2] + \
               (m.x[3] - 10)**2 + 4*(m.x[4] - 5)**2 + (m.x[5] - 3)**2 + \
               2*(m.x[6]-1)**2 + 5*m.x[7]**2 + 7*(m.x[8] - 11)**2 + \
                2*(m.x[9] - 10)**2 + (m.x[10] - 7)**2
    model.obj = pyo.Objective(rule = o)
    
    ins = model.create_instance(data)
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    solver.solve(ins)
    
    return ins

# data = {None: {
                  
#                 }
#           }
# ins = centralised_P2(data)

# for i in range(10):
#     print('x['+str(i+1)+']: ', pyo.value(ins.x[i+1]))
# print('Objective: ', pyo.value(ins.obj))
 

def SP1_P2(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.z[1]**2 + m.z[2]**2 + m.z[1]*m.z[2] - 14*m.z[1] - 16*m.z[2])/4 + \
                    (m.x[3] - 10)**2 + 4*(m.x[4] - 5)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g4(m):
            return 3*(m.z[1] - 2)**2 + 4*(m.z[2] - 3)**2 + 2*m.x[3]**2 - 7*m.x[4] - 120 <= 0
        model.ineq4 = pyo.Constraint(rule = g4)

        def g5(m):
            return 5*m.z[1]**2 + 8*m.z[2] + (m.x[3] - 6)**2 - 2*m.x[4] - 40 <= 0
        model.ineq5 = pyo.Constraint(rule = g5)
        
    else:
        def o(m):
            return (m.x[1]**2 + m.x[2]**2 + m.x[1]*m.x[2] - 14*m.x[1] - 16*m.x[2])/4 + \
                    (m.x[3] - 10)**2 + 4*(m.x[4] - 5)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g4(m):
            return 3*(m.x[1] - 2)**2 + 4*(m.x[2] - 3)**2 + 2*m.x[3]**2 - 7*m.x[4] - 120 <= 0
        model.ineq4 = pyo.Constraint(rule = g4)

        def g5(m):
            return 5*m.x[1]**2 + 8*m.x[2] + (m.x[3] - 6)**2 - 2*m.x[4] - 40 <= 0
        model.ineq5 = pyo.Constraint(rule = g5)
    

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
    
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins

def SP2_P2(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.z[1]**2 + m.z[2]**2 + m.z[1]*m.z[2] - 14*m.z[1] - 16*m.z[2])/4 + \
                    (m.x[5] - 3)**2 + 2*(m.x[6] - 1)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g6(m):
            return m.z[1]**2 + 2*(m.z[2] - 2)**2 - 2*m.z[1]*m.z[2] + 14*m.x[5] - 6*m.x[6] <= 0
        model.ineq6 = pyo.Constraint(rule = g6)
    
        def g7(m):
            return 0.5*(m.z[1] - 8)**2 + 2*(m.z[2] - 4)**2 + 3*m.x[5]**2 - m.x[6] - 30 <= 0
        model.ineq7 = pyo.Constraint(rule = g7)
        
    else:
        def o(m):
            return (m.x[1]**2 + m.x[2]**2 + m.x[1]*m.x[2] - 14*m.x[1] - 16*m.x[2])/4 + \
                    (m.x[5] - 3)**2 + 2*(m.x[6] - 1)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g6(m):
            return m.x[1]**2 + 2*(m.x[2] - 2)**2 - 2*m.x[1]*m.x[2] + 14*m.x[5] - 6*m.x[6] <= 0
        model.ineq6 = pyo.Constraint(rule = g6)
    
        def g7(m):
            return 0.5*(m.x[1] - 8)**2 + 2*(m.x[2] - 4)**2 + 3*m.x[5]**2 - m.x[6] - 30 <= 0
        model.ineq7 = pyo.Constraint(rule = g7)

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
       
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins


def SP3_P2(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.z[1]**2 + m.z[2]**2 + m.z[1]*m.z[2] - 14*m.z[1] - 16*m.z[2])/4 + \
                    5*m.x[7]**2 + 7*(m.x[8] - 11)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g1(m):
            return -105 + 4*m.z[1] + 5*m.z[2] - 3*m.x[7] + 9*m.x[8] <= 0
        model.ineq1 = pyo.Constraint(rule = g1)

        def g2(m):
            return 10*m.z[1] - 8*m.z[2] - 17*m.x[7] + 2*m.x[8] <= 0
        model.ineq2 = pyo.Constraint(rule = g2)
        
    else:
        def o(m):
            return (m.x[1]**2 + m.x[2]**2 + m.x[1]*m.x[2] - 14*m.x[1] - 16*m.x[2])/4 + \
                    5*m.x[7]**2 + 7*(m.x[8] - 11)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g1(m):
            return -105 + 4*m.x[1] + 5*m.x[2] - 3*m.x[7] + 9*m.x[8] <= 0
        model.ineq1 = pyo.Constraint(rule = g1)

        def g2(m):
            return 10*m.x[1] - 8*m.x[2] - 17*m.x[7] + 2*m.x[8] <= 0
        model.ineq2 = pyo.Constraint(rule = g2)

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
       
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins
    
def SP4_P2(data, return_solver = False):
    model = pyo.AbstractModel()
    #model.x_init = pyo.Param()
    model.z_I = pyo.Set()
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)
    model.rho = pyo.Param()
    model.I = pyo.Set()
    model.x = pyo.Var(model.I, within = pyo.Reals, bounds = (-10, 10))

    if return_solver:
        def o(m):
            return (m.z[1]**2 + m.z[2]**2 + m.z[1]*m.z[2] - 14*m.z[1] - 16*m.z[2])/4 + \
                    2*(m.x[9] - 10)**2 + (m.x[10] - 7)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g3(m):
            return -8*m.z[1] + 2*m.z[2] + 5*m.x[9] - 2*m.x[10] - 12 <= 0
        model.ineq3 = pyo.Constraint(rule = g3)

        def g8(m):
            return -3*m.z[1] + 6*m.z[2] + 12*(m.x[9] - 8)**2 - 7*m.x[10] <= 0
        model.ineq8 = pyo.Constraint(rule = g8)
        
    else:
        def o(m):
            return (m.x[1]**2 + m.x[2]**2 + m.x[1]*m.x[2] - 14*m.x[1] - 16*m.x[2])/4 + \
                    2*(m.x[9] - 10)**2 + (m.x[10] - 7)**2 + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.z_I)
        model.obj = pyo.Objective(rule = o)
        
        def g3(m):
            return -8*m.x[1] + 2*m.x[2] + 5*m.x[9] - 2*m.x[10] - 12 <= 0
        model.ineq3 = pyo.Constraint(rule = g3)

        def g8(m):
            return -3*m.x[1] + 6*m.x[2] + 12*(m.x[9] - 8)**2 - 7*m.x[10] <= 0
        model.ineq8 = pyo.Constraint(rule = g8)

    ins = model.create_instance(data)
    
    #ins.x = ins.x_init
       
    #solver = pyo.SolverFactory('cbc')
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(ins)
    
    if return_solver:
        return ins, res
    else:
        return ins


def f1(z_list, rho, global_ind, index, u_list = None, solver = False):

    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return SP1_P2(data, return_solver = solver)

def f2(z_list, rho, global_ind, index, u_list = None, solver = False):

    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0

    return SP2_P2(data, return_solver = solver)

def f3(z_list, rho, global_ind, index, u_list = None, solver = False):

    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0

    return SP3_P2(data, return_solver = solver)

def f4(z_list, rho, global_ind, index, u_list = None, solver = False):

    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index}
                }
          }
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0

    return SP4_P2(data, return_solver = solver)


# rho = 1000
# N_it = 500

# N = 2
# N_var = 3
# list_fi = [f1, f2, f3, f4]

# global_ind = [1, 2]
# index_agents = {1: [1, 2, 3, 4], 2: [1, 2, 5, 6], 
#                 3: [1, 2, 7, 8], 4: [1, 2, 9, 10]}
# z = {1: 0, 2:0}
# z_list = {1: [2.1719963529282182], 2: [2.3636830445489374]}

# actual_f = 13.864179350870021
# actual_x = 0.398

# ins1 = f1(z_list, rho, global_ind, index_agents[1], solver = True)
# ins2 = f2(z_list, rho, global_ind, index_agents[2], solver = True)
# ins3 = f3(z_list, rho, global_ind, index_agents[3], solver = True)
# ins4 = f4(z_list, rho, global_ind, index_agents[4], solver = True)




