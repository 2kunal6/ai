import torch

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0] # celcius
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] #
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_u = 0.1 * t_u # normalize

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

w = torch.ones(())
b = torch.zeros(())
t_p = model(t_u, w, b)
loss = loss_fn(t_p, t_c)
print(loss)


'''
the derivative loss function is basically using infinitesimally small delta for the following equations
delta = 0.1
# how much does the loss changes as w changes a little in value
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w

loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)
b = b - learning_rate * loss_rate_of_change_b
'''
# this function returns a tensor which represents how much the loss would change if we slightly vary the prediction
def derviative_loss_fn(t_p, t_c):
    # the following is the derivative of (t_predicted(i) - t_actual(i)) ** 2 for i = 1 to n
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

# apply the derivatives to the model
def model_derivative_of_weights(t_u, w, b):
    return t_u
def model_derivative_of_bias(t_u, w, b):
    return 1.0

def gradient_function(t_u, t_c, t_p, w, b):
    dloss_dtp = derviative_loss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * model_derivative_of_weights(t_u, w, b)
    dloss_db = dloss_dtp * model_derivative_of_bias(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = gradient_function(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params

trained_params = training_loop(n_epochs = 5000, learning_rate = 1e-4, params = torch.tensor([1.0, 0.0]), t_u = t_u, t_c = t_c)
print(trained_params)