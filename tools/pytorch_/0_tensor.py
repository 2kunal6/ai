import torch

basic_tensor = torch.ones(5)
print(basic_tensor)
basic_tensor[2] = 100.11

print(basic_tensor)

tensor_from_list = torch.tensor([1.0, 3.0, 1.0, 2.0, 6.0])
print(tensor_from_list)

two_d_tensor = torch.tensor([[1, 2], [4,1], [100, 10]])
print(two_d_tensor)
print(two_d_tensor.shape)
print(two_d_tensor[2, 0])
print(two_d_tensor[1])
print(torch.transpose(two_d_tensor, 0, 1))


n_d_tensor = torch.zeros(5, 2, 3, 3)
'''print(n_d_tensor)
print(n_d_tensor[4:])
print(n_d_tensor[:2])
print(n_d_tensor[1::3])
print(n_d_tensor[4:, 1])'''
print(n_d_tensor.dtype)
print(n_d_tensor.storage())

# broadcasting
t1 = torch.tensor([[1, 2, 3]])
print(t1+10)

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[10], [20], [30]])
print(t1 *t2)

t_3x1 = torch.tensor([[1], [2], [3]])
print(t_3x1)
t_1x3 = torch.tensor([[10, 20, 30]])
print(t_1x3)
print(t_3x1 * t_1x3)