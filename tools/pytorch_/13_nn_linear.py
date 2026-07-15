import torch
import torch.nn as nn
from torch import optim

X = torch.ones(10, 1)

linear_model = nn.Linear(1, 1)
out = linear_model(X)
print(out)
print(linear_model.weight)
print(linear_model.bias)

# ---------

t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
print(t_c)
print(t_c.shape)
t_c = torch.tensor(t_c).unsqueeze(1) # add batch
t_u = torch.tensor(t_u).unsqueeze(1) # add batch
print(t_c)
print(t_c.shape)


def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()

        optimizer.step()
        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}, Validation loss {loss_val.item():.4f}")


n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]
val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
print(list(linear_model.parameters()))

training_loop(3000, optimizer, linear_model, nn.MSELoss(), train_t_u, val_t_u, train_t_c, val_t_c)

print(linear_model.weight)
print(linear_model.bias)