import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        identity = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x + identity


class DPCNN(nn.Module):
    def __init__(self, ecfp_input, maccs_input, hidden_dim, conv_channel, output_dim, task='classification'):
        super().__init__()
        self.ecfp_input = ecfp_input
        self.maccs_input = maccs_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task

        self.ecfp_encoder = nn.Linear(ecfp_input, hidden_dim)
        self.maccs_encoder = nn.Linear(maccs_input, hidden_dim)

        self.res_layer1 = ResBlock(in_channels=1, out_channel=conv_channel, kernel_size=(3, 3), padding='same')
        self.res_layer2 = ResBlock(conv_channel, conv_channel*2, (3, 3), padding='same')

        self.fc1 = nn.Linear(conv_channel*2 * hidden_dim**2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        ecfp_input = x["ecfp"]
        maccs_input = x["maccs"]

        # 1. Augmented outer product
        ecfp = self.ecfp_encoder(ecfp_input).unsqueeze(2)
        macc = self.maccs_encoder(maccs_input).unsqueeze(1)

        # unit_tensor = torch.Tensor([[1]] * len(ecfp)) # (batch_size, 1)

        # augmented_ecfp = torch.cat([ecfp, unit_tensor], dim=1)
        # augmented_macc = torch.cat([macc, unit_tensor], dim=1)
        
        matrix = torch.matmul(ecfp, macc).unsqueeze(1)

        # 2. Residual Block
        out = self.res_layer1(matrix)
        out = self.res_layer2(out)

        # 3. Flatten and Predict
        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = F.relu(self.bn1(out))

        out = self.fc2(out)
        out = F.relu(self.bn2(out))

        logit = self.out(out)

        if self.task=='classification':
            logit = torch.sigmoid(logit)

        return logit