import sys
from DPCNN import DPCNN
from CustomDataset import FpDataset
from utils.splitter import scaffold_split, random_scaffold_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchmetrics import AUROC
import pandas as pd

import time




def time_check(total_time):
    H = 60*60
    M = 60
    if total_time >= H:
        return f"{int(total_time//H)}h {(int(total_time-H)//M)}m {total_time%M:.2f}s"
    elif total_time >= M:
        return f"{(int(total_time)//M)}m {total_time%M:.2f}s"
    else:
        return f"{total_time%M:.2f}s"



def eval(model, val_dl, loss_func):
    model.eval()
    torch.cuda.empty_cache()
    val_eval.reset()
    val_loss = 0.0

    for data in val_dl:
        x = {"ecfp": data["ecfp"].to(device), "maccs": data["maccs"].to(device)}
        y = data["y"].to(device)
        pred = model(x).squeeze()
        loss = loss_func(input=pred, target=y)
        val_eval.update(preds=pred, target=y)
        val_loss += loss.item() * len(data)
        
    val_score = val_eval.compute()
    val_loss = val_loss/len(val_dl)
    return val_score, val_loss


task = "classification" # "classification" or "regression"

data = pd.read_csv("./myexp/DPCNN/opcnn/LightBBB_dataset/y_test_indices.csv")
dataset = FpDataset(data,"BBB")
train_idx, val_idx, test_idx = scaffold_split(dataset=dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

batch_size = 64*2
train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPCNN(2048, 166, 100, 32, 1).to(device)
epoch = 100
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

if task == "classification":
    loss_func = nn.BCELoss()
elif task == "regression":
    loss_func = nn.MSELoss()

train_eval = AUROC(task="binary")
val_eval = AUROC(task="binary")

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
        train_eval.update(preds=pred, target=y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(data)

    train_score = train_eval.compute()
    train_loss = train_loss/len(train_dl)

    print(f"Epoch {_e+1}\nTrain AUROC: {train_score:.5f}, Train loss: {train_loss:.5f}")

    val_score, loss_ = eval(model, val_dl, loss_func)
    print(f"Valid AUROC: {val_score:.5f}, Valid loss: {loss_:.5f}")

end_time = time.time()
total_time = end_time - start_time

print(f"done {time_check(total_time)}")