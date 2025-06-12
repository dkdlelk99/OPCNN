import argparse
from DPCNN import DPCNN
from GCNN import GCN
from CustomDataset import FpDataset, GraphDataset
from utils.splitter import scaffold_split, random_scaffold_split
from utils.etc import time_check

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchmetrics import AUROC
import pandas as pd
import time


task = "regression" # "classification" or "regression"



def eval(model, val_dl, val_dataset, loss_func):
    model.eval()
    torch.cuda.empty_cache()
    val_loss = 0.0

    for data in val_dl:
        x = {"ecfp": data["ecfp"].to(device), "maccs": data["maccs"].to(device)}
        y = data["y"].to(device)
        pred = model(x).squeeze()
        loss = loss_func(input=pred, target=y)
        val_loss += loss.item() * len(y)
        
    val_loss = val_loss/len(val_dataset)
    return val_loss




data = pd.read_csv("./logpData/data/drugbank/drugbank_approved_logP_smi.csv") # /myPath/workspace/opcnn
dataset = FpDataset(data, "logP")

print(f"Number of logP data: {len(dataset)}")

train_idx, val_idx, test_idx = random_scaffold_split(dataset=dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

batch_size = 2**8
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if task == "classification":
    loss_func = nn.BCELoss()
    distill_loss_func = nn.KLDivLoss(reduction='batchmean', log_target=True)
elif task == "regression":
    loss_func = nn.MSELoss()
    distill_loss_func = nn.MSELoss(reduction='mean')


"""
ooooooooooooo                     o8o                   ooooooooo.                      
8'   888   `8                     `"'                   `888   `Y88.                    
     888      oooo d8b  .oooo.   oooo  ooo. .oo.         888   .d88'  .ooooo.   .oooo.o 
     888      `888""8P `P  )88b  `888  `888P"Y88b        888ooo88P'  d88' `88b d88(  "8 
     888       888      .oP"888   888   888   888        888`88b.    888ooo888 `"Y88b.  
     888       888     d8(  888   888   888   888        888  `88b.  888    .o o.  )88b 
    o888o     d888b    `Y888""8o o888o o888o o888o      o888o  o888o `Y8bod8P' 8""888P' 
"""


val_scores = []
test_scores = []

epoch = 800

model = DPCNN(2048, 166, 100, 32, 1, task='regression').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) #  weight_decay=0.0001

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params:,}")


start_time = time.time()

for _e in range(epoch):
    model.train()

    train_loss = 0.0

    for data in train_dl:
        x = {"ecfp": data["ecfp"].to(device), "maccs": data["maccs"].to(device)}
        y = data["y"].to(device)
        optimizer.zero_grad()
        
        pred = model(x).squeeze()
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y)

    train_loss = train_loss/len(train_dataset)

    print(f"Epoch {_e+1}\nTrain loss: {train_loss:.5f}")

    loss_ = eval(model, val_dl, val_dataset, loss_func)
    val_scores.append(loss_)
    print(f"Valid loss: {loss_:.5f}")

    test_loss = eval(model, test_dl, test_dataset, loss_func)
    test_scores.append(test_loss)



end_time = time.time()
total_time = end_time - start_time

print(f"done {time_check(total_time)}")


print(f"Test socre: {test_scores[val_scores.index(min(val_scores))]:.5f}, \
      when val score is {min(val_scores):.5f}, epoch: {val_scores.index(min(val_scores)):.5f}")


# parser = argparse.ArgumentParser()
# parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
# args = parser.parse_args()

