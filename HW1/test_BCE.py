# BCE

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

net = MLP(
    linear_1_in_features=4,
    linear_1_out_features=32,
    f_function='relu',
    linear_2_in_features=32,
    linear_2_out_features=1,
    g_function='sigmoid'
)
x = torch.randn(10, 4)
y = ((torch.randn(10) > 0.5) * 1.0).unsqueeze(-1)

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = bce_loss(y, y_hat)
net.backward(dJdy_hat)

#------------------------------------------------
# check the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(4, 32)),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(32, 1))
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = torch.sigmoid(net_autograd(x))

J_autograd = F.binary_cross_entropy(y_hat_autograd, y)
#print(y_hat, y_hat_autograd)
#print(J, J_autograd)
net_autograd.zero_grad()
J_autograd.backward()

print('dJdW1', net.grads['dJdW1'])
print(net_autograd.linear1.weight.grad.data)
print('dJdb1', net.grads['dJdb1'])
print(net_autograd.linear1.bias.grad.data)
print('dJdW2', net.grads['dJdW2'])
print(net_autograd.linear2.weight.grad.data)
print('dJdb2', net.grads['dJdb2'])
print(net_autograd.linear2.bias.grad.data)

print((net_autograd.linear1.weight.grad.data - net.grads['dJdW1']).norm())
print((net_autograd.linear1.bias.grad.data - net.grads['dJdb1']).norm())
print((net_autograd.linear2.weight.grad.data - net.grads['dJdW2']).norm())
print((net_autograd.linear2.bias.grad.data - net.grads['dJdb2']).norm())

#------------------------------------------------
