import torch
import numpy as np
from problems.sp_problem import *

def mape(y_true, y_pred, epsilon=1e-10):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def wmape(y_true, y_pred, epsilon = 1e-10):
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    weights = torch.abs(y_true) / torch.sum(torch.abs(y_true)) 
    weighted_mape = torch.sum(torch.abs((y_true - y_pred) / (y_true + epsilon)) * weights * 100).to("cpu")
    return weighted_mape

class Regret():
    def __init__(self, opt_pro, device):
        self.opt_pro = opt_pro
        self.device = device
        self.type = self.opt_pro.get_params()['opt_type']

    def multi_samples_solves(self, labels_true, labels_pred, add):
        self.sol = self.opt_pro(labels_pred[0], labels_true[0], add[0]).unsqueeze(0)
        for i in range(1, labels_pred.shape[0]):
            self.sol = torch.cat((self.sol, self.opt_pro(labels_pred[i], labels_true[i], add[i]).unsqueeze(0)), dim=0) 
        return self.sol
        
    def __call__(self, y_true, y_pred, add):
        x_pred = self.multi_samples_solves(y_pred, y_true, add).to(torch.float32)
        x_true = self.multi_samples_solves(y_true, y_true, add).to(torch.float32)
        sample_num = y_pred.shape[0]
        
        task_pred = torch.zeros(sample_num)
        task_true = torch.zeros(sample_num)
        for i in range(sample_num):
            if self.type == "KS":
                task_pred[i] = y_true[i, :10].t() @ x_true[i]
                task_true[i] = y_true[i, :10].t() @ x_pred[i]

            if self.type == "AP":
                task_pred[i] = y_true[i, :10].t() @ x_true[i]
                task_true[i] = y_true[i, :10].t() @ x_pred[i]

            if self.type == "SP":
                num_samples = 1
                dim = y_true.shape[-1]
                c = y_true[i, :]

                _, _, edges_list = get_A_b(dim)
                num_edges = len(edges_list)

                c = y_true[0:num_samples, :, :]
                c = c.reshape(num_samples, -1)
                c_new = torch.zeros((num_samples, 2 * num_edges)).to(self.device)
                for k, e in enumerate(edges_list):
                    x, y = e
                    c_new[:, k] = c[:, x] 
                    c_new[:, num_edges+k] = c[:, y] 
                task_pred[i] = c_new[0, :].t() @ x_pred[i]
                task_true[i] = c_new[0, :].t() @ x_true[i]

            if self.type == "TT":
                timestamp = self.opt_pro.get_params()['timestamp']
                beta = self.opt_pro.get_params()['beta']
                num_trains = self.opt_pro.get_params()['num_trains']
                A_true = x_true[i, 0:180]
                D_true = x_true[i, 180:360]
                V_true = x_true[i, 360:364]
                A_pred = x_pred[i, 0:180]
                D_pred = x_pred[i, 180:360]
                V_pred = x_pred[i, 360:364]

                task_true[i] = sum(A_true[t] - D_true[t] for t in range(timestamp)) + beta * sum(V_true[k] for k in range(num_trains))
                task_pred[i] = sum(A_true[t] - D_pred[t] for t in range(timestamp)) + beta * sum(V_pred[k] for k in range(num_trains))

            if self.type == "PF":
                label_true = torch.squeeze(y_true[i])
                task_pred[i] = (1/2) * torch.matmul(x_pred[i].T, torch.matmul(add[i], x_pred[i])) + label_true.t() @ x_pred[i]
                task_true[i] = (1/2) * torch.matmul(x_true[i].T, torch.matmul(add[i], x_true[i])) + label_true.t() @ x_true[i]

        return wmape(task_true, task_pred)