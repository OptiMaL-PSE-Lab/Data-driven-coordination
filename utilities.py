# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:22:39 2022

@author: dv516
"""


import numpy as np

def postprocessing(ax1, ax2, string, result, actual_f, coord_input = False, ALADIN = False, BO = False, samecoord = False, N=2):
    if BO:
        obj_global = result
        conv_arr = (obj_global - actual_f)**2 
        n_eval = np.arange(len(obj_global))+1
        ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
        ax2.step(n_eval, obj_global, '--', where = 'post', label = string)
    elif ALADIN:
        if samecoord:
            obj_arr = np.array(result.best_obj)
            n_eval = np.array(result.n_eval)
            obj_global = np.array(result.obj_global)
            # z_arr = np.array(result.center_list)
            # ax2.scatter(z_arr[:,0], z_arr[:,1], label = string, s = 10)
            conv_arr = (obj_arr - actual_f)**2 
            ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
            ax2.step(n_eval, obj_arr, '--', where = 'post', label = string)
        else:
            
            obj_global = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
            # z_arr1 = np.mean([result.z_list[idx+1][global_ind[0]] for idx in range(N)], axis = 0)
            # z_arr2 = np.mean([result.z_list[idx+1][global_ind[1]] for idx in range(N)], axis = 0)
            # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)
            conv_arr = (obj_global - actual_f)**2 
            n_eval = np.arange(len(obj_global))+1
            ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
            ax2.step(n_eval, obj_global, '--', where = 'post', label = string)
    elif not coord_input:
        obj_arr = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
        # z_arr1 = np.array(result.z_list[global_ind[0]])
        # z_arr2 = np.array(result.z_list[global_ind[1]])
        conv_arr = (obj_arr - actual_f)**2 
        ax1.step(np.arange(len(obj_arr))+1, conv_arr, '--', where = 'post', label = string)
        # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)   
        ax2.step(np.arange(len(obj_arr))+1, obj_arr, '--', where = 'post', label = string)
    else:
        # f = np.array(result['f_store'])
        # x_list = np.array(result['x_store'])
        # x_best = np.array(result['x_best_so_far'])
        f_best = np.array(result['f_best_so_far'])
        ind_best = np.array(result['samples_at_iteration'])       
        ax1.step(ind_best, (f_best - actual_f)**2, '--', where = 'post', label = string)
        # ax2.plot(x_best[:,0], x_best[:,1], '--', c = 'k', linewidth = 1)
        # ax2.scatter(x_list[:,0], x_list[:,1], label = string, s = 10)
        ax2.step(ind_best, f_best, '--', where = 'post', label = string)
        
    return ax1, ax2

# def draw_Test1(ax, string, result, actual_f, coord_input = False, ALADIN = False, BO = False, samecoord = False, N=2):
#     if BO:
#         obj_global = result
#         conv_arr = (obj_global - actual_f)**2 
#         n_eval = np.arange(len(obj_global))+1
#         ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
#         ax2.step(n_eval, obj_global, '--', where = 'post', label = string)
#     elif ALADIN:
#         if samecoord:
#             obj_arr = np.array(result.best_obj)
#             n_eval = np.array(result.n_eval)
#             obj_global = np.array(result.obj_global)
#             # z_arr = np.array(result.center_list)
#             # ax2.scatter(z_arr[:,0], z_arr[:,1], label = string, s = 10)
#             conv_arr = (obj_arr - actual_f)**2 
#             ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
#             ax2.step(n_eval, obj_arr, '--', where = 'post', label = string)
#         else:
            
#             obj_global = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
#             # z_arr1 = np.mean([result.z_list[idx+1][global_ind[0]] for idx in range(N)], axis = 0)
#             # z_arr2 = np.mean([result.z_list[idx+1][global_ind[1]] for idx in range(N)], axis = 0)
#             # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)
#             conv_arr = (obj_global - actual_f)**2 
#             n_eval = np.arange(len(obj_global))+1
#             ax1.step(n_eval, conv_arr, '--', where = 'post', label = string)
#             ax2.step(n_eval, obj_global, '--', where = 'post', label = string)
#     elif not coord_input:
#         obj_arr = np.sum(np.array([result.obj[i+1] for i in range(N)]), axis = 0)
#         # z_arr1 = np.array(result.z_list[global_ind[0]])
#         # z_arr2 = np.array(result.z_list[global_ind[1]])
#         conv_arr = (obj_arr - actual_f)**2 
#         ax1.step(np.arange(len(obj_arr))+1, conv_arr, '--', where = 'post', label = string)
#         # ax2.scatter(z_arr1, z_arr2, label = string, s = 10)   
#         ax2.step(np.arange(len(obj_arr))+1, obj_arr, '--', where = 'post', label = string)
#     else:
#         # f = np.array(result['f_store'])
#         # x_list = np.array(result['x_store'])
#         # x_best = np.array(result['x_best_so_far'])
#         f_best = np.array(result['f_best_so_far'])
#         ind_best = np.array(result['samples_at_iteration'])       
#         ax1.step(ind_best, (f_best - actual_f)**2, '--', where = 'post', label = string)
#         # ax2.plot(x_best[:,0], x_best[:,1], '--', c = 'k', linewidth = 1)
#         # ax2.scatter(x_list[:,0], x_list[:,1], label = string, s = 10)
#         ax2.step(ind_best, f_best, '--', where = 'post', label = string)
        
#     return ax

def construct_A(index_dict, global_list, N_ag, only_global = False):
    N_g = len(global_list)
    if not only_global:
        A_list = {i+1: np.zeros(((N_ag-1)*N_g, len(index_dict[i+1]))) for i in range(N_ag)}
        for i in range(len(global_list)):
            for j in range(N_ag-1):
                idx = np.where(np.array(index_dict[j+1]) == np.array(global_list[i]))
                A_list[j+1][i*(N_ag-1)+j, idx] = 1
                idx = np.where(np.array(index_dict[j+2]) == np.array(global_list[i]))
                A_list[j+2][i*(N_ag-1)+j, idx] = -1
    else:
        A_list = {i+1: np.zeros(((N_ag-1)*N_g, N_g)) for i in range(N_ag)}
        for i in range(len(global_list)):
            for j in range(N_ag-1):
                A_list[j+1][i*(N_ag-1)+j, i] = 1
                A_list[j+2][i*(N_ag-1)+j, i] = -1
    return A_list

def preprocess_BO(array, init):
    f_best = array.copy()
    N_eval = len(f_best)
    f_best[0] = float(init)
    best = float(init)
    for j in range(1, N_eval):
        if (best > f_best[j]):
            best = f_best[j]
        else:
            f_best[j] = best

    return f_best



