# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:33:43 2021

@author: dv516
"""


import pyomo.environ as pyo
import pandas as pd

import numpy as np


def economic_Lagr(data, return_solver, dim=None):
    if dim is None:
        raise ValueError('dim is None')
        
    df_nut = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Nutritional')
    # names = df_nut['Name'].tolist()
    # nutritional_columns = df_nut.columns[1:].tolist()
    #df_nutritional[nutritional_columns[0]]
    df_econ = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Economic')
    cost = df_econ[df_econ.columns[1]].values.tolist()
    # df_env = pd.read_excel('AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Sustainable')
    df_const = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Constraints')
    nut_constr = df_const['Constraint'].tolist()
    np.random.seed(0)
    # powers = [1, 2]
    # a = np.random.randint(0, 2, len(names))
    
    model = pyo.AbstractModel()
    model.c = pyo.ConstraintList()
    # m.z_I = pyo.Set()
    # m.x = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    # m.y = pyo.Var(m.z_I, within = pyo.Binary)
    # m.z = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    model.I = pyo.Set(initialize = data[None]['I'][None])
    model.x = pyo.Var(model.I, within = pyo.NonNegativeReals)
    model.y = pyo.Var(model.I, within = pyo.Binary)
    model.z = pyo.Param(model.I, within = pyo.Reals)
    model.u = pyo.Param(model.I, within = pyo.Reals)
    # {value: z_temp[count] for count, value in enumerate(names)}
    
    m = model.create_instance(data)
    
    for count, value in enumerate(nut_constr):
        m.c.add(sum(m.x[n]*df_nut[value][n-1]/100 for n in m.I) <= df_const['Max.'][count]/100)
        m.c.add(sum(m.x[n]*df_nut[value][n-1]/100 for n in m.I) >= df_const['Min.'][count]/100)
    for n in m.I:
        m.c.add(m.x[n] <= m.y[n] )
    m.c.add(sum(m.x[n] for n in m.I) == 1)
    m.c.add(sum(m.y[n] for n in m.I) <= 6)
    def o(m):
        return sum(m.x[n]*cost[n-1] + 1e6*(m.x[n]-m.z[n]+m.u[n])**2 for n in m.I)
    m.obj = pyo.Objective(rule = o)
    
    
    # solver = pyo.SolverFactory('ipopt')
    solver = pyo.SolverFactory('mosek')
    solver.solve(m)
    
    return m


def f1(z_list, rho, global_ind, index, u_list = None, solver = False, dim=None):
            
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
    
    # print(data)
    
    return economic_Lagr(data, return_solver = solver, dim=dim)


def sust(z, dim=None):
    
    if dim is None:
        raise ValueError('dim is None')
        
    df_nut = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Nutritional')
    names = df_nut['Name'].tolist()
    # nutritional_columns = df_nut.columns[1:].tolist()
    #df_nutritional[nutritional_columns[0]]
    # df_econ = pd.read_excel('Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Economic')
    # cost = df_econ[df_econ.columns[1]].values.tolist()
    df_env = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Sustainable')
    # df_const = pd.read_excel('Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Constraints')
    # nut_constr = df_const['Constraint'].tolist()
    np.random.seed(0)
    powers = [1, 2]
    a = np.random.randint(0, 2, len(names))
    
    model = pyo.ConcreteModel()
    model.c = pyo.ConstraintList()
    # m.z_I = pyo.Set()
    # m.x = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    # m.y = pyo.Var(m.z_I, within = pyo.Binary)
    # m.z = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    model.I = pyo.RangeSet(1, 28)
    model.x = pyo.Var(model.I, within = pyo.NonNegativeReals, bounds = (0, 1))
    model.y = pyo.Var(model.I, within = pyo.Binary)
    # model.z = pyo.Param(model.I, within = pyo.Reals)
    # {value: z_temp[count] for count, value in enumerate(names)}
    # model.nbr_ag = pyo.RangeSet(1, model.N-1)
    # model.lmbda_coeff = pyo.Param(model.I, model.nbr_ag)
    # model.lmbda = pyo.Param(model.I, model.nbr_ag)
    
    def o(m):
        np.random.seed(0)
        bilinear = np.random.randint(1, len(names), 5)
        return sum((m.x[n])**powers[a[n-1]]*df_env['CC (kg CO2-eq)'][n-1] + 
                   sum((m.x[k]*100)*(m.x[k+1]*100) for k in bilinear) for n in m.I) + \
                       1e6*sum((m.x[j] - z[j-1] )**2 for j in m.I)
    model.obj = pyo.Objective(rule = o)
    
    # m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
    
    # solver = pyo.SolverFactory('ipopt')
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)
    
    return model

def sust_Lagr(data, return_solver, dim=None):
    if dim is None:
        raise ValueError('dim is None')
        
    df_nut = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Nutritional')
    names = df_nut['Name'].tolist()
    # nutritional_columns = df_nut.columns[1:].tolist()
    #df_nutritional[nutritional_columns[0]]
    # df_econ = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Economic')
    # cost = df_econ[df_econ.columns[1]].values.tolist()
    df_env = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Sustainable')
    # df_const = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Constraints')
    # nut_constr = df_const['Constraint'].tolist()
    np.random.seed(0)
    powers = [1, 2]
    a = np.random.randint(0, 2, len(names))
    
    model = pyo.AbstractModel()
    model.c = pyo.ConstraintList()
    model.rho = pyo.Param()
    # m.z_I = pyo.Set()
    # m.x = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    # m.y = pyo.Var(m.z_I, within = pyo.Binary)
    # m.z = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    model.I = pyo.Set(initialize = data[None]['I'][None])
    model.x = pyo.Var(model.I, within = pyo.NonNegativeReals, bounds = (0, 1))
    # model.y = pyo.Var(model.I, within = pyo.Binary)
    model.z = pyo.Param(model.I, within = pyo.Reals)
    model.u = pyo.Param(model.I, within = pyo.Reals)
    # {value: z_temp[count] for count, value in enumerate(names)}
    # model.nbr_ag = pyo.RangeSet(1, model.N-1)
    # model.lmbda_coeff = pyo.Param(model.I, model.nbr_ag)
    # model.lmbda = pyo.Param(model.I, model.nbr_ag)
    
    m = model.create_instance(data)
    
    def o(m):
        np.random.seed(0)
        bilinear = np.random.randint(1, len(names), 5)
        return sum((m.z[n])**powers[a[n-1]]*df_env['CC (kg CO2-eq)'][n-1] + \
                       1000000*(m.x[n]-m.z[n]+m.u[n])**2 for n in m.I) + \
                   sum((m.z[k]*100)*(m.z[k+1]*100) for k in bilinear) + \
                    m.rho/2*sum((m.x[j] - m.z[j] + m.u[j])**2 for j in m.I)
    m.obj = pyo.Objective(rule = o)
    
    # m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
    
    # solver = pyo.SolverFactory('ipopt')
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m)
    
    return m

def f2(z_list, rho, global_ind, index, u_list = None, solver = False, dim=None):
    
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
    
    return sust_Lagr(data, return_solver = solver, dim=dim)

def economic_Lagr_Lin(data, return_solver, dim=None):
    if dim is None:
        raise ValueError('dim is None')
        
    df_nut = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Nutritional')
    # names = df_nut['Name'].tolist()
    # nutritional_columns = df_nut.columns[1:].tolist()
    #df_nutritional[nutritional_columns[0]]
    df_econ = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Economic')
    cost = df_econ[df_econ.columns[1]].values.tolist()
    # df_env = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Sustainable')
    df_const = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Constraints')
    nut_constr = df_const['Constraint'].tolist()
    np.random.seed(0)
    # powers = [1, 2]
    # a = np.random.randint(0, 2, len(names))
    
    model = pyo.AbstractModel()
    model.c = pyo.ConstraintList()
    # m.z_I = pyo.Set()
    # m.x = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    # m.y = pyo.Var(m.z_I, within = pyo.Binary)
    # m.z = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    model.I = pyo.Set(initialize = data[None]['I'][None])
    model.x = pyo.Var(model.I, within = pyo.NonNegativeReals)
    model.u = pyo.Param(model.I, within = pyo.Reals)
    # model.y = pyo.Var(model.I, within = pyo.Binary)
    model.z = pyo.Param(model.I, within = pyo.Reals)
    # {value: z_temp[count] for count, value in enumerate(names)}
    
    m = model.create_instance(data)
    
    for count, value in enumerate(nut_constr):
        m.c.add(sum(m.x[n]*df_nut[value][n-1]/100 for n in m.I) <= df_const['Max.'][count]/100)
        m.c.add(sum(m.x[n]*df_nut[value][n-1]/100 for n in m.I) >= df_const['Min.'][count]/100)
    # for n in m.I:
    #     m.c.add(m.x[n] <= m.y[n] )
    m.c.add(sum(m.x[n] for n in m.I) == 1)
    # m.c.add(sum(m.y[n] for n in m.I) <= 6)
    def o(m):
        return sum(m.x[n]*cost[n-1] + 1e6*(m.x[n]-m.z[n]+m.u[n])**2 for n in m.I)
    m.obj = pyo.Objective(rule = o)
    
    
    # solver = pyo.SolverFactory('ipopt')
    solver = pyo.SolverFactory('mosek')
    solver.solve(m)
    
    return m


def f1Lin(z_list, rho, global_ind, index, u_list = None, solver = False, dim=None):
    
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
    
    return economic_Lagr_Lin(data, return_solver = solver, dim=dim)

def sust_Lin_Lagr(data, return_solver, dim=None):
    if dim is None:
        raise ValueError('dim is None')
        
    df_nut = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Nutritional')
    names = df_nut['Name'].tolist()
    # nutritional_columns = df_nut.columns[1:].tolist()
    #df_nutritional[nutritional_columns[0]]
    # df_econ = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Economic')
    # cost = df_econ[df_econ.columns[1]].values.tolist()
    df_env = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Sustainable')
    # df_const = pd.read_excel('../Problems/AnimalNutrition - ' + str(dim) + ' variables.xlsx', sheet_name = 'Constraints')
    # nut_constr = df_const['Constraint'].tolist()
    np.random.seed(0)
    powers = [1, 2]
    a = np.random.randint(0, 2, len(names))
    
    model = pyo.AbstractModel()
    model.c = pyo.ConstraintList()
    model.rho = pyo.Param()
    # m.z_I = pyo.Set()
    # m.x = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    # m.y = pyo.Var(m.z_I, within = pyo.Binary)
    # m.z = pyo.Var(m.z_I, within = pyo.NonNegativeReals)
    model.I = pyo.Set(initialize = data[None]['I'][None])
    model.x = pyo.Var(model.I, within = pyo.NonNegativeReals, bounds = (0, 1))
    # model.y = pyo.Var(model.I, within = pyo.Binary)
    model.z = pyo.Param(model.I, within = pyo.Reals)
    model.u = pyo.Param(model.I, within = pyo.Reals)
    # {value: z_temp[count] for count, value in enumerate(names)}
    # model.nbr_ag = pyo.RangeSet(1, model.N-1)
    # model.lmbda_coeff = pyo.Param(model.I, model.nbr_ag)
    # model.lmbda = pyo.Param(model.I, model.nbr_ag)
    
    m = model.create_instance(data)
    
    def o(m):
        np.random.seed(0)
        # bilinear = np.random.randint(1,len(names), 5)
        return sum((m.z[n])**powers[a[n-1]]*df_env['CC (kg CO2-eq)'][n-1] + \
                       1e6*(m.x[n]-m.z[n]+m.u[n])**2 for n in m.I) + \
                       m.rho/2*sum((m.x[j] - m.z[j]+m.u[j])**2 for j in m.I)
    m.obj = pyo.Objective(rule = o)
    
    # m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
    
    # solver = pyo.SolverFactory('ipopt')
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m)
    
    return m

def f2Lin(z_list, rho, global_ind, index, u_list = None, solver = False, dim=None):
    
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
    
    return sust_Lin_Lagr(data, return_solver = solver, dim=dim)



