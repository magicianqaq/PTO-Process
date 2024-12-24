import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import sqrt
from untils import *

class PredictModel():
    def __init__(self, p_pre, p_set):
        self.params = p_pre
        self.type = p_pre['pre_type']
        self.seed = p_set['seed']
                
    def __call__(self, trainset):
        fix_seed(self.seed)
        features, labels, adds = trainset[0]
        input_dim=features.shape[-1]
        output_dim=labels.shape[-1]
        if self.type == "MLP":
            self.size = self.params['size']
            model = MLP(input_dim, output_dim, self.size)
            
        elif self.type == "CNN":
            self.in_channels = self.params['in_channels']
            model = CombRenset18(input_dim, output_dim, self.in_channels)
            
        elif self.type == "LSTM":
            self.hidden_dim = self.params['hidden_dim']
            self.num_layers = self.params['num_layers']
            model = LSTMRegressor(input_dim, output_dim, self.hidden_dim, self.num_layers)
            
        else:
            model = None
            print("No predictive model is available")
           
        return model

class MLP(nn.Module):
    def __init__(self, input_size, output_size, size):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(input_size)
        self.linear1 = nn.Linear(input_size, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return x

class CombRenset18(nn.Module):

    def __init__(self, input_size, out_features, in_channels=3):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (out_features, out_features)
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        return x

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, bidirectional=True, dropout_prob=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_prob if num_layers > 1 else 0)

        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_multiplier, output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: shape (batch_size, seq_length, input_size)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        out = self.dropout(hidden)
        out = self.fc(out)
        out = out.unsqueeze(1)
        return out