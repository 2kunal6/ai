import numpy as np
import csv
import torch
import torch.nn.functional as F

wine_path = "resources/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
print(wineq_numpy)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(col_list)

wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape)

X = wineq[:, :-1] # or wineq[:, :(len(wineq)-1)]
y = wineq[:, -1]

print(X.shape)
print(y.shape)

y_onehot = F.one_hot(y.long(), 10)
print(y_onehot[:15])
print(y[:15])

X_mean = torch.mean(X, dim=0)
X_variance = torch.var(X, dim=0)
X_normalized = (X - X_mean) / torch.sqrt(X_variance)
