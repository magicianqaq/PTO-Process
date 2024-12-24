import numpy as np
import torch
import os
from problems.sp_problem import *
from solvers.QP import *
from solvers.MILP import *
from problems.timetable_problem import *

class OptProblem():
    def __init__(self, p_opt, p_set, device):
        self.params = p_opt
        self.type = p_opt['opt_type']
        self.device = device
        # labels shape: (samples, label)
        if self.type == "SP":
            self.sol = SP(self.params, device)
        elif self.type == "PF":
            self.sol = PF(self.params, device)
        elif self.type == "KS":
            self.sol = KS(self.params, device)
        elif self.type == "AP":
            self.sol = AP(self.params, device)
        elif self.type == "TT":
            self.sol = TT(self.params, device)
        else:
            self.model = None
            print("No opt model is available")
        
    def __call__(self, label_true, label_pre, add):
        solution = self.sol(label_true, label_pre, add)
        return solution

    def get_params(self):
        return self.params
    
    def get_device(self):
        return self.device

class KS():
    def __init__(self, params, device):
        self.cap = params['cap']
        self.int_type = params['int_type']
        self.verbose = params['verbose']
        self.device = device

    def __call__(self, label_true, label_pred, add):
        label_pred = label_pred.to("cpu").detach().numpy()
        label_true = label_true.to("cpu").detach().numpy()
        
        label_pred1 = label_pred[0:10]
        label_pred2 = label_pred[10:20]
        label_true1 = label_true[0:10]
        label_true2 = label_true[10:20]
        num_x = len(label_pred1)
        
        c_p = label_pred1     
        A_p = label_pred2 
        b_p = np.array([self.cap])
        x_1, obj_val = solve_mip_max(A_p, b_p, c_p, int_type=self.int_type)
        
        if np.array_equal(label_true, label_pred) != True:
            c = np.squeeze(label_true1)            
            A_a = np.expand_dims((label_true2), axis=0)
            A = np.concatenate((A_a, np.eye(num_x)), axis=0)
            b = np.concatenate((b_p, x_1))
            x_opt, obj_val = solve_mip_max(A, b, c, x_1, int_type=self.int_type)
        else:
            x_opt = x_1

        x_opt = torch.from_numpy(x_opt).to(self.device)
        return x_opt

class AP():
    def __init__(self, params, device):
        self.req = params['req']
        self.int_type = params['int_type']
        self.verbose = params['verbose']
        self.device = device

    def __call__(self, label_true, label_pred, add):
        label_pred = label_pred.to("cpu").detach().numpy()
        label_true = label_true.to("cpu").detach().numpy()

        label_pred1 = np.squeeze(label_pred[0:10])
        label_pred2 = np.squeeze(label_pred[10:20])
        label_true1 = np.squeeze(label_true[0:10])
        label_true2 = np.squeeze(label_true[10:20])
        num_x = len(label_pred1)

        c_p = np.squeeze(label_pred1)         
        A1 = np.expand_dims(np.squeeze(label_pred2), axis=0)
        A2 = np.eye(num_x)
        b1 = np.array([self.req])
        b2 = np.zeros(num_x)
        A_p = - np.concatenate((A1, A2), axis=0)
        b_p = - np.concatenate((b1, b2))
        x_1, obj_val = solve_mip(A_p, b_p, c_p)

        if np.array_equal(label_true, label_pred) != True:
            c_p = np.squeeze(label_true1)           
            A1 = np.expand_dims(np.squeeze(label_true2), axis=0)
            A2 = np.eye(num_x)
            b1 = np.array([self.req])
            b2 = x_1
            A_p = - np.concatenate((A1, A2), axis=0)
            b_p = - np.concatenate((b1, b2))
            x_opt, obj_val = solve_mip(A_p, b_p, c_p, x_1)
        else:
            x_opt = x_1

        x_opt = torch.from_numpy(x_opt).to(self.device)
        return x_opt

class SP():
    def __init__(self, params, device):
        self.device = device

    def __call__(self, label_true, label_pred, add):
        label_pred = torch.unsqueeze(label_pred, 0)
        label_true = torch.unsqueeze(label_true, 0)
        label_pred = label_pred.to("cpu").detach().numpy()
        label_true = label_true.to("cpu").detach().numpy()
        # 18 * 18
        num_samples = 1
        dim = label_pred.shape[-1]
        A, b, edges_list  = get_A_b(dim)

        num_edges = len(edges_list)
        num_vertexs = dim * dim

        c = label_pred[0:num_samples, :, :]
        c = c.reshape(num_samples, -1)
        c_new = np.zeros((num_samples, 2 * num_edges))
        for i, e in enumerate(edges_list):
            x, y = e
            c_new[:, i] = c[:, x] 
            c_new[:, num_edges+i] = c[:, y] 
    
        _, _, solution = lpm.ComputeBasis(c=c_new, A=A, b=b)
        # print("Shape: ", c_new.shape, solution.shape)
        x_opt = torch.squeeze(torch.from_numpy(solution).to(self.device))
        return x_opt

class TT():
    def __init__(self, params, device):
        self.num_trains = params['num_trains']
        self.timestamp = params['timestamp']
        self.beta = params['beta']
        self.device = device

    def __call__(self, label_true, label_pred, add):
        label_pred = torch.squeeze(label_pred).to("cpu").detach().numpy()
        label_true = torch.squeeze(label_true).to("cpu").detach().numpy()
        # 6
        num_x = self.num_trains
        
        x_opt = np.zeros(num_x)
        z_1 = np.zeros((num_x, self.timestamp))
        A_1, D_1, _, z_1, _, _, _, _ = timetable_problem(labels=label_pred, num_trains=self.num_trains, beta=self.beta, get_all=True)

        A = np.zeros(len(A_1))
        D = np.zeros(len(D_1))

        # solutions for second stage
        A, D, V, _ = timetable_problem2(labels=label_true, num_trains=self.num_trains, x_1=np.squeeze(z_1), beta=self.beta)
        x_opt = np.concatenate((A, D, V))
        x_opt = torch.squeeze(torch.from_numpy(x_opt).to(self.device))
        return x_opt

class PF():
    def __init__(self, params, device):
        self.alpha = params['alpha']
        self.device = device

    def __call__(self, label_true, label_pred, add):
        label_pred = label_pred.to("cpu").detach().numpy()
        add = add.to("cpu").detach().numpy()
        # 50 * 1
        num_x = len(label_pred[0])
        con_matrixs = add

        Q = 2 * self.alpha * con_matrixs
        c = - np.squeeze(label_pred)      
        G = - np.eye(num_x) 
        h = np.zeros(num_x)         
        A = np.full(num_x, 1)      
        b = np.array([1])        
        x_opt, _ = solve_qp(Q, c, G, h, A, b)

        x_opt = torch.squeeze(torch.from_numpy(x_opt).to(self.device))
        return x_opt