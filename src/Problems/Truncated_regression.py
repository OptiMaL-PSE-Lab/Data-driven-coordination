# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:27:06 2021

@author: dv516
"""

import pyomo.environ as pyo

def regression(data):
    model = pyo.AbstractModel()
    
    model.I = pyo.Set()
    model.N_data = pyo.Set()
    
    model.x = pyo.Var(model.I, within=pyo.Reals)
    model.loss = pyo.Var(within=pyo.Reals)
    
    # model.z = pyo.Param(model.I,        within=pyo.Reals)
    model.y = pyo.Param(model.N_data,   within=pyo.Reals)
    model.H = pyo.Param(model.N_data, model.I, within=pyo.Reals)
    model.xi = pyo.Param()
    model.N_all = pyo.Param()
    model.N = pyo.Param()
    model.rho_reg = pyo.Param()
    
    # def loss(m):
    #     return m.loss == rho_reg/2/N_data*sum(pyo.log(1+sum((m.y[n] - m.x[i]*m.H[n,i])**2 for i in m.I)) for n in m.N_data)
    # model.loss_constr = pyo.Constraint(rule=loss)
    def o(m):
        return m.rho_reg/2/(m.N_all/m.N)*sum(pyo.log(1+sum((m.y[n] - m.x[i]*m.H[n,i])**2/m.rho_reg for i in m.I)) for n in m.N_data) + \
                 m.xi*sum(abs(m.x[i]) for i in m.I)
    model.obj = pyo.Objective(rule=o)
    
    m = model.create_instance(data)
    
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m)
    
    return m

def regression_local(data, return_solver=True):
    model = pyo.AbstractModel()
    
    model.rho = pyo.Param()
    
    model.I = pyo.Set()
    model.N_data = pyo.Set()
    
    model.x = pyo.Var(model.I, within=pyo.Reals)
    model.loss = pyo.Var(within=pyo.Reals)
    
    model.u = pyo.Param(model.I)
    model.z = pyo.Param(model.I, within=pyo.Reals)
    model.y = pyo.Param(model.N_data, within=pyo.Reals)
    model.H = pyo.Param(model.N_data, model.I, within=pyo.Reals)
    model.xi = pyo.Param()
    model.N_all = pyo.Param()
    model.N = pyo.Param()
    model.rho_reg = pyo.Param()
    
    if not return_solver:
        def o(m):
            return m.rho_reg/2/(m.N_all/m.N)*sum(pyo.log(1+sum((m.y[n] - m.x[i]*m.H[n,i])**2/m.rho_reg for i in m.I)) for n in m.N_data) + \
                    m.xi/2*sum(abs(m.x[i]) for i in m.I) + \
                    m.rho*sum((m.x[i] - m.z[i] + m.u[i])**2 for i in m.I)
        model.obj = pyo.Objective(rule=o)
        
    else:
        def o(m):
            return m.rho_reg/2/(m.N_all/m.N)*sum(pyo.log(1+sum((m.y[n] - m.z[i]*m.H[n,i])**2/m.rho_reg for i in m.I)) for n in m.N_data) + \
                    m.xi/2*sum(abs(m.z[i]) for i in m.I) + \
                    m.rho*sum((m.x[i] - m.z[i] + m.u[i])**2 for i in m.I)
        model.obj = pyo.Objective(rule=o)
    
    m = model.create_instance(data)
    
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m)
    
    return m

def f(z_list, rho, global_ind, index, data_big, idx, N_all, N, u_list = None, 
      solver = False, rho_reg = 3, xi = 0.01):
    
    data = {None: {
                  'z': {}, 'u': {}, 'z_I': {None: global_ind}, 
                  'rho': {None: rho}, 'I': {None: index},
                  'N_data': {}, 'y': {}, 'H': {}, 'xi': {None: xi},
                  'N_all': {None: N_all}, 'N': {None: N},
                  'rho_reg': {None: rho_reg},
                }
          }
    
    y = data_big[idx]['y']
    H = data_big[idx]['H']
    N_data, dim = H.shape
    
    data[None]['N_data'][None] = [n+1 for n in range(N_data)]
    
    for i in range(dim):
        for n in range(N_data):
            data[None]['H'][(n+1,i+1)] = float(H[n,i])
    for n in range(N_data): 
        data[None]['y'][n+1] = float(y[n])
    
    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return regression_local(data, return_solver = solver)


