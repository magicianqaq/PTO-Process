from metric import *
import numpy as np
import torch
from torch.utils.data import Subset
import os

def fix_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def dataset_split(dataset, train_rate):
    # data split
    train_size = int(train_rate * len(dataset))

    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, test_dataset


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_save = None
        self.state = False

    def __call__(self, val_loss, model, epoch, epoch_num):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter % 10 == 0:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}, ({epoch}/{epoch_num})'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
            self.best_score = score
            self.counter = 0
        

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...'
            )

        subdirectory = 'data'
        file_name = 'best_model.pth'
        full_path = os.path.join(subdirectory, file_name)

        torch.save(model.state_dict(), full_path)


class multisolvers():
    def __init__(self, opt_pro):
        self.opt_pro = opt_pro
        self.params = self.opt_pro.get_params()
        self.type = self.params['opt_type']
        self.device = self.opt_pro.get_device()

    def get_params(self):
        return self.params
    
    def get_device(self):
        return self.device

    def multi_samples_solves(self, labels_true, labels_pred, add):
        self.sol = self.opt_pro(labels_pred[0], labels_true[0], add[0]).unsqueeze(0)
        for i in range(1, labels_pred.shape[0]):
            self.sol = torch.cat((self.sol, self.opt_pro(labels_pred[i], labels_true[i], add[i]).unsqueeze(0)), dim=0) 
        return self.sol

    def __call__(self, labels_true, labels_pred, add):
        x = self.multi_samples_solves(labels_true, labels_pred, add).to(torch.float32)
        return x


class get_obj():
    def __init__(self, opt_pro):
        self.opt_pro = opt_pro
        self.type = self.opt_pro.get_params()['opt_type']

    def get_obj(self, y_true, y_pred, x_true, x_pred, add):
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
 
        return task_pred, task_true

    def __call__(self, labels_true, labels_pred, add):
        x = self.multi_samples_solves(labels_true, labels_pred, add).to(torch.float32)
        return x
