from torch.utils.data import dataloader
from metric import *
from pto import *
import numpy as np
import torch
import time
import os
import sys
from untils import *


class Train():
    def __init__(self, p_tr, p_set, p_ds, device):
        self.type = p_set['pto_file']
        self.warm_start = p_set['warm_start']

        self.seed = p_tr['seed']
        self.lr = p_tr['lr']
        self.epoch_num = p_tr['epoch_num']
        self.patience = p_tr['patience']
        self.verbose = p_tr['verbose']

        self.train_rate_pre = p_ds['train_rate_pre']
        self.batch_size_pre = p_ds['batch_size_pre']

        self.device = device

    def __call__(self, trainset, model, check_point):
        device = self.device
        epoch_num = self.epoch_num
        fix_seed(self.seed)

        train_loader = DataLoader(trainset, batch_size=self.batch_size_pre)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss().to(device)
        print('Init over')
        print('Train start')
        # train and validate
        if self.warm_start != 0:
            for epoch in range(0, epoch_num):
                # model train
                tr_loss = 0
                model.train()
                for tr in enumerate(train_loader):
                    i_batch, (tr_X, tr_Y, add) = tr
                    target = model(tr_X)
                    loss = mse(input=tr_Y, target=target)
                    tr_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if epoch > 0 and epoch % 100 == 0:
                    print(f'epoch: {epoch}')
                if epoch <= 100:
                    check_point(model, must_check=True)
                else:
                    check_point(model)
            print('Warm start over')
            
        return model


class OptTrain():
    def __init__(self, p_set, p_pto, p_ds, device):
        self.p_set = p_set
        self.p_pto = p_pto
        self.p_ds = p_ds

        self.type = p_set['pto_file']
        self.best_model_path = p_set['best_model_path']
        self.seed = p_set['seed']

        self.train_rate_loss = p_ds['train_rate_loss']
        self.batch_size_loss = p_ds['batch_size_loss']

        self.device = device

    def __call__(self, train_dataset, model, solver, check_point):

        solvers = multisolvers(solver)

        # labels shape: (samples, label)
        print('The predict then optimize method used is: ' + self.type)
        if self.type == "2STG":
            return model
            
        elif self.type == "SPO":
            self.pred_opt = SPOplus(self.p_pto, self.p_ds, solvers, self.device)
            
        elif self.type == "IntOpt":
            self.pred_opt = IntOpt(self.p_pto, solver, self.device)
            
        elif self.type == "DBB":
            self.pred_opt = DBB(self.p_pto, self.p_ds, solvers, self.device)
            
        elif self.type == "LODLs":
            self.pred_opt = LODLs(self.p_pto, self.p_ds, solvers, self.device)
            
        elif self.type == "LANCER":
            self.pred_opt = Lancer(self.p_pto, self.p_ds, solvers, self.device)
            
        elif self.type == "Ours":
            self.pred_opt = Ours(self.p_pto, self.p_ds, solvers, self.device)

        else:
            print("No trainer is available")

        strat_time = time.time()

        fix_seed(self.seed)

        # optimize the model
        print('PTO start')
        model = self.pred_opt(train_dataset, model, check_point)

        torch.save(model.state_dict(), self.best_model_path)

        end_time = time.time()
        print("Pto running time:", end_time - strat_time)

        check_point.get_result()
        return model


class Test():
    def __init__(self, p_te, p_set, device):
        
        self.device = device
        self.best_model_path = p_set['best_model_path']
        
    def __call__(self, test_dataset, model, opt):

        self.regret = Regret(opt, self.device)

        batch_size = len(test_dataset)

        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # test 
        test_wmape = 0
        test_regret = 0
        for te in enumerate(test_loader):
            model.eval()
            i_batch, (te_X, te_Y, add) = te

            te_X, te_Y = te_X.to(self.device), te_Y.to(self.device)
            
            target = model(te_X)

            test_wmape += wmape(y_true=te_Y, y_pred=target)
            test_regret += self.regret(y_true=te_Y, y_pred=target, add=add)
            
        avg_wmape = test_wmape/len(test_loader)
        show_wmape = f"{avg_wmape:.3g}%"
        print("The prediction error is: " + show_wmape)

        avg_regret = test_regret/len(test_loader)
        show_regret = f"{avg_regret:.3g}%"
        print("The decision error is: " + show_regret)


class CheckPoint():
    def __init__(self, dataset, p_set, device, solver):
        self.device = device
        self.type = p_set['pto_file']
        self.time_limit = p_set['time_limit']
        self.process_type = p_set['process_type']
        self.interval = p_set['time_limit'] / 100


        self.len_dataset = len(dataset)
        batch_size = len(dataset)
        self.loader = DataLoader(dataset, batch_size = batch_size)
        self.regret = Regret(solver, device)
        self.result_dec = {}
        self.result_pre = {}
        self.time_process = 0
        self.last_time = time.time()

    def __call__(self, model, must_check=False):

        now_time = time.time()
        if now_time - self.last_time <= self.interval and must_check == False:
            return None
        self.time_process += now_time - self.last_time
        time_str = f"{self.time_process:.3g}s"
        print("The time point is :" + time_str)

        print("check now")
        test_regret = 0
        test_wmape =0
        for te in enumerate(self.loader):
            model.eval()
            i_batch, (te_X, te_Y, add) = te

            te_X, te_Y = te_X.to(self.device), te_Y.to(self.device)
            
            target = model(te_X)

            test_wmape += wmape(y_true=te_Y, y_pred=target)
            test_regret += self.regret(y_true=te_Y, y_pred=target, add=add)
           
        avg_regret = test_regret/len(self.loader)
        Regret_str = f"{avg_regret:.3g}%"

        avg_wmape = test_wmape/len(self.loader)
        show_wmape = f"{avg_wmape:.3g}%"

        print("The prediction error is: " + show_wmape)     
        print("The decision error is: " + Regret_str)

        self.save_result(avg_regret, avg_wmape, self.time_process)
        self.last_time = time.time()
        self.get_result()
        if self.time_process >= self.time_limit:
            print("The process prediction result is: ", self.result_pre)
            print("The process decision result is: ", self.result_dec)
            sys.exit("Reach the time limit")


    def save_result(self, Regret_val, Predict_val, time_val):
        self.result_dec[time_val] = Regret_val
        self.result_pre[time_val] = Predict_val


    def get_result(self):
        result_path = r".\result\\time_regret\\"

        with open(result_path + self.type + f'_process_{self.len_dataset}_{self.process_type}_pre_result.pkl', 'wb') as f:
            pickle.dump(self.result_pre, f)

        with open(result_path + self.type + f'_process_{self.len_dataset}_{self.process_type}_dec_result.pkl', 'wb') as f:
            pickle.dump(self.result_dec, f)


