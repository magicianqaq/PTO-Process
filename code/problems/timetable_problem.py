# -*- coding: utf-8 -*-
from logging.handlers import DatagramHandler
from ssl import HAS_NEVER_CHECK_COMMON_NAME
import numpy as np
import pandas as pd
import torch
from gurobipy import *
import matplotlib.pyplot as plt
import matplotlib
import pickle
from types import SimpleNamespace
matplotlib.use('TkAgg')

def just_for_timetable(params, flow_sample, run_time_path, one=False):
    timetable = {}
    if one:
        flow_sample = [flow_sample]
    for i in range(len(flow_sample)):
        timetable[i] = normal_model(params, flow_sample[i], run_time_path, purpose='timetable')
    return timetable


def get_all(params, flow, run_time_path, one=False):
    timetable = {}
    obj = {}
    if one:
        flow = [flow]
    for i in range(len(flow)):
        timetable[i], obj[i] = normal_model(params, flow[i], run_time_path, purpose='all')
    return timetable, obj

def get_params(params=None):
    if params == None:
        params_ass = {}
        params_ass['Num_stations'] = 3
        params_ass['H_l'] = 12
        params_ass['H_u'] = 48
        params_ass['T_l'] = 2
        params_ass['T_u'] = 12
        params_ass['C'] = 100
        params_ass['T_safe'] = 11
        params_ass['seed'] = 42
        params_ass['Passenger_end_step'] = 120
        params_ass['end_step'] = 6
        params_ass['time_granularity'] = 5 
        params_ass = SimpleNamespace(**params_ass)
        return params_ass
    else:
        return params


def timetable_problem(labels, num_trains, beta, params=None, get_all=False):
    # -*- coding: utf-8 -*-
    if params == None:
        params = get_params(params)
    H_l = params.H_l
    H_u = params.H_u
    C = params.C
    np.random.seed(params.seed)
    time_granularity = params.time_granularity
    end_time = params.end_step * time_granularity * 6

    '''------创建模型------'''
    model = Model('TT')

    '''------数据导入------'''
    # 导入OD数据
    n_p = np.zeros(end_time)
    labels = labels
    for i in range(len(labels)):
        num_passengers = labels[i].astype(int)
        if num_passengers == 0:
                continue
        timestamps = np.random.uniform(end_time//len(labels) * i, end_time//len(labels) * (i+1), num_passengers)
        rounded_times = np.ceil(timestamps - 1)
        integer_times = rounded_times.astype(int)
        unique, counts = np.unique(integer_times, return_counts=True)
        for t in range(len(unique)):
                n_p[unique[t]] = counts[t]
    P = n_p
    A = np.zeros(end_time)
    for t in range(end_time):
        A[t] = sum(n_p[t1] for t1 in range(t))

    '''------定义决策变量------'''
    I = model.addVars(num_trains, vtype=GRB.INTEGER, name="I", lb=0, ub=C) # 上车人数
    z = model.addVars(num_trains, end_time, vtype=GRB.BINARY, name="z") # 离开状态

    '''------定义中间变量------'''
    D = model.addVars(end_time, vtype=GRB.INTEGER, name="D", lb=0)  # 累积离开人数
    TD = model.addVars(num_trains, vtype=GRB.INTEGER, name="TD", lb=0) # 发车时间
    V = model.addVars(num_trains, vtype=GRB.INTEGER, name="V", lb=0) # 滞留人数
    H = model.addVars(num_trains, vtype=GRB.INTEGER, name="H", lb=H_l, ub=H_u) # 发车间隔
   
    '''------边界约束条件------'''
    # 边界约束
    model.addConstr(TD[num_trains - 1] == end_time - 1, name='TD_constr0')
    model.addConstr(D[0] == 0, name='D_constr0')
    model.addConstr((V[0] == quicksum((1 - z[0, t]) * P[t] 
                     for t in range(end_time)) - I[0]), name="V_constr0")
    model.addConstr(H[0] == TD[0], name='H_constr0')

    model.addConstrs((z[k, 0] == 0 for k in range(num_trains)), name='z_constr0')


    '''------中间变量定义约束条件------'''
    model.addConstrs((D[t] == D[t-1] + quicksum(I[k] * (z[k, t] - z[k, t-1]) for k in range(num_trains)) 
                      for t in range(1, end_time)), name='D_constr1')
    model.addConstrs((H[k] == TD[k] - TD[k-1] for k in range(1, num_trains)), name='H_constr1')
    model.addConstrs((V[k] == V[k-1] + quicksum((z[k-1, t] - z[k, t]) * P[t] for t in range(end_time)) - I[k] 
                      for k in range(1, num_trains)), name="V_constr1")
    model.addConstrs((TD[k] == quicksum(t * (z[k, t] - z[k, t-1]) for t in range(1, end_time)) 
                      for k in range(num_trains)), name="TD_constr1")
    
    '''------约束条件------'''
    #　状态变量自约束
    model.addConstrs((z[k, t-1] <= z[k, t] 
                      for k in range(num_trains) for t in range(1, end_time)), name='z_constr1')

    '''------目标函数------'''
    obj = quicksum(A[t] - D[t] for t in range(end_time)) + beta * quicksum(V[k] for k in range(num_trains))

    '''------求解模型------'''
    model.setObjective(obj, GRB.MINIMIZE)
    model.setParam('outputflag', 0)
    # model.setparam(grb.param.timelimit, 30)
    model.optimize()
    
    if get_all:
        D = np.array([D[t].X for t in range(end_time)])
        I = np.array([I[k].X for k in range(num_trains)])
        z = np.array([[z[k, t].X for t in range(end_time)] for k in range(num_trains)])
        TD = np.array([TD[k].X for k in range(num_trains)])
        H = np.array([H[k].X for k in range(num_trains)])
        V = np.array([V[k].X for k in range(num_trains)])
        return A, D, I, z, TD, H, V, model.objval 
    else:
        timetable = np.array([TD[k].X for k in range(num_trains)])
        return timetable, model.objval


def timetable_problem2(labels, num_trains, beta, x_1 ,params=None):
    # -*- coding: utf-8 -*-
    if params == None:
        params = get_params(params)
    H_l = params.H_l
    H_u = params.H_u
    C = params.C
    np.random.seed(params.seed)
    time_granularity = params.time_granularity
    end_time = params.end_step * time_granularity * 6

    '''------创建模型------'''
    model = Model('TT')

    '''------数据导入------'''
    # 导入OD数据
    n_p = np.zeros(end_time)
    labels = labels
    for i in range(len(labels)):
        num_passengers = labels[i].astype(int)
        if num_passengers == 0:
                continue
        timestamps = np.random.uniform(end_time//len(labels) * i, end_time//len(labels) * (i+1), num_passengers)
        rounded_times = np.ceil(timestamps - 1)
        integer_times = rounded_times.astype(int)
        unique, counts = np.unique(integer_times, return_counts=True)
        for t in range(len(unique)):
                n_p[unique[t]] = counts[t]
    P = n_p
    A = np.zeros(end_time)
    for t in range(end_time):
        A[t] = sum(n_p[t1] for t1 in range(t))

    '''------定义决策变量------'''
    I = model.addVars(num_trains, vtype=GRB.INTEGER, name="I", lb=0, ub=C) # 上车人数
    z = x_1 # 离开状态

    '''------定义中间变量------'''
    D = model.addVars(end_time, vtype=GRB.INTEGER, name="D", lb=0)  # 累积离开人数
    V = model.addVars(num_trains, vtype=GRB.INTEGER, name="V", lb=0) # 滞留人数
   
    '''------边界约束条件------'''
    # 边界约束
    model.addConstr(D[0] == 0, name='D_constr0')
    model.addConstr((V[0] == quicksum((1 - z[0, t]) * P[t] 
                     for t in range(end_time)) - I[0]), name="V_constr0")

    '''------中间变量定义约束条件------'''
    model.addConstrs((D[t] == D[t-1] + quicksum(I[k] * (z[k, t] - z[k, t-1]) for k in range(num_trains)) 
                      for t in range(1, end_time)), name='D_constr1')
    model.addConstrs((V[k] == V[k-1] + quicksum((z[k-1, t] - z[k, t]) * P[t] for t in range(end_time)) - I[k] 
                      for k in range(1, num_trains)), name="V_constr1")
    
    '''------约束条件------'''
    #　状态变量自约束

    '''------目标函数------'''
    obj = quicksum(A[t] - D[t] for t in range(end_time)) + beta * quicksum(V[k] for k in range(num_trains))

    '''------求解模型------'''
    model.setObjective(obj, GRB.MINIMIZE)
    model.setParam('outputflag', 0)
    # model.setparam(grb.param.timelimit, 30)
    model.optimize()

    D = np.array([D[t].X for t in range(end_time)])
    V = np.array([V[k].X for k in range(num_trains)])
    return A, D, V, model.objval 
