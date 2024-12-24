from numpy._typing import _ObjectCodes
from metric import *
import numpy as np
import torch
import time
from problems.sp_problem import *
from torch.utils.data import DataLoader, Dataset
from untils import *
import torch.nn as nn
from typing import Union

Activation = Union[str, nn.Module]

class SPOplus():
    def __init__(self, p_pto, p_ds, solver, device):
        self.type = p_pto['type']
        self.lr = p_pto['lr']
        self.epoch_num = p_pto['epoch_num']

        self.train_rate_pto = p_ds['train_rate_pto']
        self.batch_size_pto = p_ds['batch_size_pto']

        self.lp = SP_SPO.apply
        self.solver = solver

        self.device = device
        
    def criterion(self, x_pred, x_target):
        return (0.5*((x_target - x_pred)**2).sum())/ x_target.shape[0]

    def __call__(self, train_dataset, model, check_point):

        device = self.device
        batch_size = int(len(train_dataset) / 10)
        # batch_size = self.batch_size_pto
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        for get in enumerate(train_loader):
            idx, (x, y_true, adds) = get


        x = x.to(device)
        y_true = y_true.to(device)
        adds = adds.to(device)
        z_true = self.solver(y_true, y_true, adds).to(device)

        print('X shape: ', x.shape)
        print('Y shape: ', y_true.shape)
        print('Z shape: ', z_true.shape)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        print('Init over')
        check_point(model)

        # mse = torch.nn.MSELoss().to(device)
        # optimize train
        for epoch in range(self.epoch_num):
            tr_loss = 0
            model.train()
            y_pred = model(x).to(device)
            z_pred = self.lp(y_pred, y_true, adds, self.solver, device)
            loss = self.criterion(z_pred, z_true)
            tr_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            check_point(model)

        return model 


class DBB():
    def __init__(self, p_pto, p_ds, solver, device):
        self.type = p_pto['type']
        self.lr = p_pto['lr']
        self.epoch_num = p_pto['epoch_num']
        self.lambda_val = p_pto['lambda_val']

        self.train_rate_pto = p_ds['train_rate_pto']
        self.batch_size_pto = p_ds['batch_size_pto']

        self.lp = SP_DBB.apply
        self.solver = solver

        self.device = device
        
    def criterion(self, z_pred, z_target):
        return (0.5*((z_target - z_pred)**2).sum())/ z_target.shape[0]

    def __call__(self, train_dataset, model, check_point):

        device = self.device
        batch_size = int(len(train_dataset) / 10)
        # batch_size = self.batch_size_pto
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        for get in enumerate(train_loader):
            idx, (x, y_true, adds) = get

        x = x.to(device)
        y_true = y_true.to(device)
        adds = adds.to(device)
        z_true = self.solver(y_true, y_true, adds).to(device)

        print('X shape: ', x.shape)
        print('Y shape: ', y_true.shape)
        print('Z shape: ', z_true.shape)

        dataset = MyDataset(x, z_true, adds)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # mse = torch.nn.MSELoss().to(device)

        check_point(model)
        # optimize train
        for epoch in range(self.epoch_num):
            for tr in enumerate(train_loader):
                i_batch, (x, z_true, adds) = tr
                tr_loss = 0
                model.train()
                y_pred = model(x).to(device)
                z_pred = self.lp(y_pred, self.lambda_val, adds, self.solver, device)
                loss = self.criterion(z_pred, z_true)
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            check_point(model)

        return model 

#　端到端层的设计，是SPO算法的关键
class SP_SPO(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y_pred, y_true, adds, solver, device):
        ctx.weights = 2*y_pred - y_true
        ctx.device = device
        zero = torch.zeros_like(ctx.weights)
        ctx.weights = torch.maximum(ctx.weights, zero)
        ctx.z_pred = solver(ctx.weights, ctx.weights, adds)
        return ctx.z_pred

    @staticmethod
    def backward(ctx, grad_output):
        c_new_grad = grad_output
        num_samples = grad_output.shape[0]
        dim = 18
        A, b, edges_list  = get_A_b(dim)
        c_grad= get_c(c_new_grad, edges_list, num_samples, dim, ctx.device)
        return -c_grad, None, None, None, None

class SP_DBB(torch.autograd.Function):
    """ Define a new function for shortest path to avoid the legacy autograd
    error """

    @staticmethod
    def forward(ctx, y_pred, lambda_val, adds, solver, device):
        ctx.weights = y_pred
        ctx.solver = solver
        ctx.device = device
        ctx.lambda_val = lambda_val
        ctx.dim = y_pred.shape[-1]
        ctx.save_for_backward(y_pred)
        ctx.adds = adds
        ctx.z_pred = solver(ctx.weights, ctx.weights, adds)
        return ctx.z_pred


    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        num_samples = grad_output.shape[0]
        A, b, edges_list  = get_A_b(ctx.dim)
        c_grad_output= get_c(grad_output, edges_list, num_samples, ctx.dim, ctx.device)

        weights_prime = weights + ctx.lambda_val * c_grad_output
        zero = torch.zeros_like(weights_prime)
        weights_prime = torch.maximum(weights_prime, zero)
        
        better_paths = ctx.solver(weights_prime, weights_prime, ctx.adds)
        gradient = (ctx.z_pred - better_paths) / ctx.lambda_val

        c_gradient= get_c(gradient, edges_list, num_samples, ctx.dim, ctx.device)
        return -c_gradient, None, None, None, None

def get_c(z, edges_list, num_samples, dim, device):

    c = torch.zeros((num_samples, dim * dim)).to(device)
    for i in range(num_samples): 
        for idx, e in enumerate(edges_list):
            x, y = e
            c[:, x] = z[:, idx]
    c = c.reshape((num_samples, dim, dim))
        
    return c


class LODLs():
    def __init__(self, p_pto, p_ds, solver, device):
        self.p_pto = p_pto
        self.type = p_pto['type']
        self.lr = p_pto['lr']
        self.epoch_num = p_pto['epoch_num']

        self.loss_epoch_num = p_pto['loss_epoch_num']
        self.loss_lr = p_pto['loss_lr']
        self.num_samples = p_pto['num_samples']
        self.params = p_pto

        self.p_ds = p_ds
        self.train_rate_pto = p_ds['train_rate_pto']
        self.batch_size_pto = p_ds['batch_size_pto']

        self.train_rate_loss = p_ds['train_rate_loss']
        self.batch_size_loss = p_ds['batch_size_loss']

        self.solver = solver
        self.device = device
        
    def __call__(self, train_dataset, model, check_point):
        # train loss
        # y_pred, loss = samples
        # y is feature, loss is target, model is your setting
        # y is the feature(X), loss is the label(Y), model is the custom model，and learned loss is the loss function
        Ys, opt_objectives, Yhats, objectives = self.get_samples(train_dataset, self.device, self.num_samples, self.solver)
        learned_loss = self.learned_loss_train(Ys, opt_objectives, Yhats, objectives, self.params)
        check_point(model)

        model = self.pto_train(train_dataset, learned_loss, model, self.params, self.device, check_point)
        return model 


    def get_samples(self, train_dataset, device, num_samples, solver):
        print("sample now")
        # input train dataset, output samples dataset(like train_dataset)
        Y_hats = []
        Ys = []
        opt_objectives = []
        objectives = []
        for i in range(len(train_dataset)):
            x, y_true, adds = train_dataset[i]
            y_std = 1e-5
            y_noise = torch.distributions.Normal(0, y_std).sample((num_samples, *y_true.shape)).to(device)
            y_hat = (y_true + y_noise)
            Y_hats.append(y_hat)
            Ys.append(y_true)
            x_true = solver(y_true.unsqueeze(0), y_true.unsqueeze(0), adds.unsqueeze(0)).to(torch.float32)
            for j in range(num_samples):
                opt_objective, objective = get_obj(y_true.unsqueeze(0), y_hat[j].unsqueeze(0), x_true, adds.unsqueeze(0), solver)
                objectives.append(objective[j])
                if j == 0:
                    opt_objectives.append(opt_objective)
        
        Ys = torch.stack(Ys).to(device)
        Y_hats = torch.stack(Y_hats).to(device)
        opt_objectives = torch.stack(opt_objectives).to(device)
        objectives = torch.stack(objectives).to(device)

        print("sample done")
    
        print(Ys.shape, objectives.shape, Y_hats.shape, opt_objectives.shape)
        return Ys, opt_objectives, Y_hats, objectives


    def learned_loss_train(self, Ys, opt_objectives, Yhats, objectives, params):
        print("learn loss now")
        # print(Ys.shape, objectives.shape, Yhats.shape, opt_objectives.shape)
        # input samples dataset, output learned loss(like mse loss)
        losses = []
        for i in range(Ys.shape[0]):
            learned_loss_one_sample = self._learned_loss_one_sample(Ys[i], opt_objectives[i], Yhats[i], objectives[i], params)
            losses.append(learned_loss_one_sample)  
        print("learn loss done")
        return losses


    def _learned_loss_one_sample(self, Y, opt_objective, Yhat, objective, params):
        lr = params['loss_lr']
        num_iters = params['loss_epoch_num']
        # one sample case
        # model: input Yhat, output objective
        """
        Function that learns a model to approximate the behaviour of the
        'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
        """
        # Load a model
        model = WeightedMSE(Y)

        # Fit a model to the points
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for iter_idx in range(num_iters):
            # Define update step using "closure" function
            def loss_closure():
                optimizer.zero_grad()
                pred = model(Yhat).flatten()
                if not (pred >= -1e-3).all().item():
                    print(f"WARNING: Prediction value < 0: {pred.min()}")
                loss = MSE(pred, objective)
                loss.backward()
                return loss

            # Make an update step
            optimizer.step(loss_closure)

        def surrogate_decision_quality(y_pred, y_true):
            return model(y_pred) - opt_objective
        return surrogate_decision_quality


    def pto_train(self, train_dataset, learned_loss, model, params, device, check_point):
        print("train now")
        batch_size = params['batch_size']
        lr = params['lr']
        epoch_num = params['epoch_num']
        loss_num = len(train_dataset)
        # input train dataset, learned loss, model, output trained model
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        # print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # train and validate
        for epoch in range(0, epoch_num):
            # model train
            tr_loss = 0
            model.train()
            for tr in enumerate(train_loader):
                losses = []
                i_batch, (tr_X, tr_Y, add) = tr
                tr_X, tr_Y = tr_X.to(device), tr_Y.to(device)
                pred = model(tr_X)
                for i in range(loss_num):
                    losses.append(learned_loss[i](pred, tr_Y))
                loss = torch.stack(losses).mean()  
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0 and epoch > 0:
                print(f'epoch {epoch}:')
                check_point(model)

        print("train done")
        return model

def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()


class WeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSE, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * self.weights.clamp(min=self.min_val)).mean(dim=-1)

        return weighted_mse


def get_obj(y_true, y_pred, x_true, add, solvers):
    x_pred = solvers(y_pred, y_true, add).to(torch.float32)
    obj_y, obj_y_hat = obj(y_true, y_pred, x_true, x_pred, add, solvers)
    return obj_y, obj_y_hat


def obj(y_true, y_pred, x_true, x_pred, add, solvers):
    p_opt = solvers.get_params()
    device = solvers.get_device()
    get_type = p_opt['opt_type']

    sample_num = y_pred.shape[0]
        
    task_pred = torch.zeros(sample_num)
    task_true = torch.zeros(sample_num)
    for i in range(sample_num):
        if get_type == "KS":
            task_pred[i] = y_true[i, :10].t() @ x_true[i]
            task_true[i] = y_true[i, :10].t() @ x_pred[i]

        if get_type == "AP":
            task_pred[i] = y_true[i, :10].t() @ x_true[i]
            task_true[i] = y_true[i, :10].t() @ x_pred[i]

        if get_type == "SP":
            num_samples = 1
            dim = y_true.shape[-1]
            c = y_true[i, :]

            _, _, edges_list = get_A_b(dim)
            num_edges = len(edges_list)

            c = y_true[0:num_samples, :, :]
            c = c.reshape(num_samples, -1)
            c_new = torch.zeros((num_samples, 2 * num_edges)).to(device)
            for k, e in enumerate(edges_list):
                x, y = e
                c_new[:, k] = c[:, x] 
                c_new[:, num_edges+k] = c[:, y] 
            task_pred[i] = c_new[0, :].t() @ x_pred[i]
            task_true[i] = c_new[0, :].t() @ x_true[i]

        if get_type == "TT":
            timestamp = p_opt['timestamp']
            beta = p_opt['beta']
            num_trains = p_opt['num_trains']
            A_true = x_true[i, 0:180]
            D_true = x_true[i, 180:360]
            V_true = x_true[i, 360:364]
            A_pred = x_pred[i, 0:180]
            D_pred = x_pred[i, 180:360]
            V_pred = x_pred[i, 360:364]

            task_true[i] = sum(A_true[t] - D_true[t] for t in range(timestamp)) + beta * sum(V_true[k] for k in range(num_trains))
            task_pred[i] = sum(A_true[t] - D_pred[t] for t in range(timestamp)) + beta * sum(V_pred[k] for k in range(num_trains))

        if get_type == "PF":
            label_true = torch.squeeze(y_true[i])
            task_pred[i] = (1/2) * torch.matmul(x_pred[i].T, torch.matmul(add[i], x_pred[i])) + label_true.t() @ x_pred[i]
            task_true[i] = (1/2) * torch.matmul(x_true[i].T, torch.matmul(add[i], x_true[i])) + label_true.t() @ x_true[i]
    return task_true, task_pred


class Lancer():
    def __init__(self, p_pto, p_ds, solver, device):
        self.p_pto = p_pto
        self.type = p_pto['type']
        self.lr = p_pto['lr']
        self.epoch_num = p_pto['epoch_num']

        self.loss_epoch_num = p_pto['loss_epoch_num']
        self.loss_lr = p_pto['loss_lr']
        self.num_samples = p_pto['num_samples']
        self.n_iter = p_pto['n_iter']
        self.params = p_pto

        self.n_layers = p_pto['n_layers']
        self.layer_size = p_pto['layer_size']
        self.max_iter = p_pto['max_iter']

        self.p_ds = p_ds
        self.train_rate_pto = p_ds['train_rate_pto']
        self.batch_size_pto = p_ds['batch_size_pto']

        self.train_rate_loss = p_ds['train_rate_loss']
        self.batch_size_loss = p_ds['batch_size_loss']

        self.solver = solver
        self.device = device
        

    def __call__(self, train_dataset, model, check_point):

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        for get in enumerate(train_loader):
            idx, (X, Y, adds) = get
        print("solve")
        Z_true = self.solver(Y, Y, adds).to(torch.float32)
        print(Z_true.shape)

        for i in range(self.n_iter):
            Y_pred, objectives = self.get_samples(X, Y, Z_true, adds, model)
            learned_loss = self.learned_loss_train(Y, Y_pred, objectives)
            check_point(model, must_check=True)

            model = self.pto_train(train_dataset, learned_loss, model, check_point, i)
            check_point(model, must_check=True)
        return model 

    def get_samples(self, X, Y, Z_true, adds, model):
        print("sample now")
        # 获取预测值
        Y_pred = model(X)
        objectives = []
        for i in range(len(X)):
            # 获取目标函数值
            opt_objective, objective = get_obj(Y[i].unsqueeze(0), Y_pred[i].unsqueeze(0), Z_true[i].unsqueeze(0), adds[i].unsqueeze(0), self.solver)
            objectives.append(objective)
        objectives = torch.stack(objectives).to(self.device)
        print("sample done")
        print(Y_pred.shape, objectives.shape)
        return Y_pred, objectives

    def learned_loss_train(self, Y, Y_pred, objectives):
        print("learn loss")
        # Load a model
        Y = torch.flatten(Y, start_dim=1)
        Y_pred = torch.flatten(Y_pred, start_dim=1)
        self.z_dim = Y.shape[-1]
        self.f_dim = objectives.shape[-1]
        self.model = MLPLancer(self.z_dim, self.f_dim, self.n_layers, self.layer_size, 'relu').to(self.device)
        N = Y.shape[0]
        batch_size = self.batch_size_loss
        n_batches = int(N / batch_size)
        total_iter = 0
        print_freq = 5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.loss_lr)
        while total_iter < self.max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                self.model.train()
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                Y_batch = Y[idxs]
                Y_pred_batch = Y_pred[idxs]
                objectives_batch = objectives[idxs]
                loss_i = self.learned_loss_one_batch(Y_batch, Y_pred_batch, objectives_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting target model C, itr: ", total_iter, ", lancer loss: ", loss_i, flush=True)
                if total_iter >= self.max_iter:
                    break
        print("learn loss done")
        return self.model


    def learned_loss_one_batch(self, Y, Y_pred, objectives):
        Y = Y.detach()
        Y_pred = Y_pred.detach()
        self.optimizer.zero_grad()
        pred = self.model(Y, Y_pred)
        loss = MSE(pred, objectives)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def pto_train(self, train_dataset, learned_loss, model, check_point, i):
        print("train now")
        batch_size = self.params['batch_size']
        lr = self.params['lr']
        epoch_num = self.params['epoch_num']
        device = self.device
        # train and validate
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        # print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train and validate
        for epoch in range(0, epoch_num):
            # model train
            tr_loss = 0
            model.train()
            for tr in enumerate(train_loader):
                losses = []
                i_batch, (tr_X, tr_Y, add) = tr
                tr_X, tr_Y = tr_X.to(device), tr_Y.to(device)
                pred = model(tr_X)
                Y = torch.flatten(tr_Y, start_dim=1)
                Y_pred = torch.flatten(pred, start_dim=1)
                loss = learned_loss(Y, Y_pred).mean()
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i <= 1:
                check_point(model, musk_check=True)
            else:
                check_point(model)
        print("train done")
        return model
    

class MLPLancer(nn.Module):
# multilayer perceptron LANCER Model
    def __init__(self,
                    z_dim,
                    f_dim,
                    n_layers,
                    layer_size,
                    out_activation="relu",
                    **kwargs):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.f_dim = f_dim
        self.n_layers = n_layers
        self.layer_size = layer_size
        #####################################
        self.model_output = build_mlp(input_size=self.z_dim,
                                                output_size=self.f_dim,
                                                n_layers=self.n_layers,
                                                size=self.layer_size,
                                                output_activation=out_activation)

    def forward(self, z_pred_tensor: torch.FloatTensor, z_true_tensor: torch.FloatTensor):
        # input = torch.cat((z_pred_tensor, z_true_tensor), dim=1)
        # input = torch.abs(z_true_tensor - z_pred_tensor)
        input = torch.square(z_true_tensor - z_pred_tensor)
        return self.model_output(input)


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        # layers.append(nn.Dropout(p=0.6))
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class Ours():
    def __init__(self, p_pto, p_ds, solver, device):
        self.p_pto = p_pto
        self.type = p_pto['type']
        self.lr = p_pto['lr']
        self.epoch_num = p_pto['epoch_num']

        self.loss_epoch_num = p_pto['loss_epoch_num']
        self.loss_lr = p_pto['loss_lr']
        self.num_samples = p_pto['num_samples']
        self.sample_lr = p_pto['sample_lr']
        self.n_iter = p_pto['n_iter']
        self.params = p_pto

        self.n_layers = p_pto['n_layers']
        self.layer_size = p_pto['layer_size']
        self.max_iter = p_pto['max_iter']

        self.p_ds = p_ds
        self.train_rate_pto = p_ds['train_rate_pto']
        self.batch_size_pto = p_ds['batch_size_pto']

        self.train_rate_loss = p_ds['train_rate_loss']
        self.batch_size_loss = p_ds['batch_size_loss']

        self.lambda_val = p_pto['lambda_val']

        self.solver = solver
        self.device = device

        self.lp = SP_DBB.apply
        
    def __call__(self, train_dataset, model, check_point):

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        for get in enumerate(train_loader):
            idx, (X, Y, adds) = get
        print("solve")
        Z_true = self.solver(Y, Y, adds).to(torch.float32)
        print(Z_true.shape)

        check_point(model, must_check=True)
        for i in range(self.n_iter):
            Y_pred, objectives, model = self.get_samples(X, Z_true, adds, model)
            check_point(model, must_check=True)

            learned_loss = self.learned_loss_train(Y, Y_pred, objectives)
            model = self.pto_train(train_dataset, learned_loss, model, check_point, i)
            check_point(model, must_check=True)
        return model 

    def get_loss_one_batch(self, z_pred, z_target):
        return 0.5*((z_target - z_pred)**2)

    def losses_mean(self, losses):
        return losses.sum() / losses.shape[0]

    def get_samples(self, X, Z_true, adds, model):
        print("sample now")
        dataset = MyDataset(X, Z_true, adds)
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # get losses
        optimizer = torch.optim.Adam(model.parameters(), lr=self.sample_lr)
        for tr in enumerate(train_loader):
            tr_loss = 0
            i_batch, (x, z_true, adds) = tr
            model.train()
            y_pred = model(x).to(self.device)
            z_pred = self.lp(y_pred, self.lambda_val, adds, self.solver, self.device)
            loss_one_batch = self.get_loss_one_batch(z_pred, z_true)
            
            if i_batch == 0:
                y_preds = y_pred
                losses = loss_one_batch
            else:
                y_preds = torch.cat((y_preds, y_pred), dim=0)
                losses = torch.cat((losses, loss_one_batch), dim=0)

            loss = self.losses_mean(loss_one_batch)
            tr_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_preds = y_preds.to(self.device)
        losses = losses.to(self.device)
        print('sample done')
        return y_preds, losses, model

    def learned_loss_train(self, Y, Y_pred, objectives):
        print("learn loss")
        # Load a model
        Y = torch.flatten(Y, start_dim=1)
        Y_pred = torch.flatten(Y_pred, start_dim=1)
        self.z_dim = Y.shape[-1]
        self.f_dim = objectives.shape[-1]
        self.model = MLPLancer(self.z_dim, self.f_dim, self.n_layers, self.layer_size, 'relu').to(self.device)
        N = Y.shape[0]
        batch_size = self.batch_size_loss
        n_batches = int(N / batch_size)
        total_iter = 0
        print_freq = 5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.sample_lr)
        while total_iter < self.max_iter:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                self.model.train()
                idxs = rand_indices[bi * batch_size: (bi + 1) * batch_size]
                Y_batch = Y[idxs]
                Y_pred_batch = Y_pred[idxs]
                objectives_batch = objectives[idxs]
                loss_i = self.learned_loss_one_batch(Y_batch, Y_pred_batch, objectives_batch)
                total_iter += 1
                if total_iter % print_freq == 0:
                    print("****** Fitting target model C, itr: ", total_iter, ", lancer loss: ", loss_i, flush=True)
                if total_iter >= self.max_iter:
                    break
        print("learn loss done")
        return self.model


    def learned_loss_one_batch(self, Y, Y_pred, objectives):
        Y = Y.detach()
        Y_pred = Y_pred.detach()
        objectives = objectives.detach()
        self.optimizer.zero_grad()
        pred = self.model(Y, Y_pred)
        loss = MSE(pred, objectives)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def pto_train(self, train_dataset, learned_loss, model, check_point, i):
        print("train now")
        batch_size = self.params['batch_size']
        lr = self.params['lr']
        epoch_num = self.params['epoch_num']
        device = self.device
        # input train dataset, learned loss, model, output trained model
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train 
        for epoch in range(0, epoch_num):
            # model train
            tr_loss = 0
            model.train()
            for tr in enumerate(train_loader):
                i_batch, (tr_X, tr_Y, add) = tr
                pred = model(tr_X)
                Y = torch.flatten(tr_Y, start_dim=1)
                Y_pred = torch.flatten(pred, start_dim=1)
                loss = learned_loss(Y, Y_pred).mean()
                tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i <= 1:
                check_point(model, must_check=True)
            else:
                check_point(model)
        print("train done")
        return model


class MyDataset(Dataset):
        def __init__(self, X, Z, adds):
            self.X = X
            self.Z = Z
            self.adds = adds

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Z[idx], self.adds[idx]