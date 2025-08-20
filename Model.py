import torch
import torch.nn as nn
import copy
import numpy as np

def make_mlp(input_dim, hidden_layers, output_dim, activation_fn: nn.Module):
    layers = []
    prev = input_dim

    for h_layer in hidden_layers:
        layers.append(nn.Linear(prev, h_layer))
        layers.append(copy.deepcopy(activation_fn))
        prev = h_layer

    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class LNN(nn.Module):
    def __init__(self, hidden_layers = (32, 64, 128, 64), activation_fn = nn.Softmax(dim = -1)):
        """
        - hidden_layers: tuple[int], e.g. [32, 64, 128, 128, 64, 32]
        - activation_fn: lambda e.g. nn.Softmax(dim=1)
        """
        super().__init__()
        # Input is [theta, theta_dot] -> 2 features; output is scalar L
        self.net = make_mlp(input_dim=2,
                            hidden_layers=hidden_layers,
                            output_dim=1,
                            activation_fn=activation_fn)

    def forward(self, theta, theta_dot):
        # I want a [N,1] for each theta and theta_dot
        state = torch.cat([theta, theta_dot], dim=-1)  # [N,2]
        return self.net(state).squeeze(-1)         # This will give out a [N]

    # def __init__(self):
    #     super(LNN, self).__init__()
    #     self.Linear_Stack = nn.Sequential(
    #         nn.Linear(2, 32),
    #         nn.Softmax(dim=1),
    #         nn.Linear(32, 64),
    #         nn.Softmax(dim=1),
    #         nn.Linear(64, 128),
    #         nn.Softmax(dim = 1),
    #         nn.Linear(128, 64),
    #         nn.Softmax(dim=1),
    #         nn.Linear(64, 1)
    #     )
    #     print(self.Linear_Stack)
    #
    # def forward(self, theta, theta_dot):
    #     state = torch.cat([theta, theta_dot], dim=-1)  # I want a [N,1] for each theta and theta_dot
    #     return self.Linear_Stack(state).squeeze(-1)  # This will give out a [N]

def LNN_Euler_Lagrange(lnn, theta, theta_dot):
    theta = theta.detach().clone().requires_grad_(True)
    theta_dot = theta_dot.detach().clone().requires_grad_(True)

    L = lnn(theta, theta_dot) # [N]

    grad_theta, grad_theta_dot = torch.autograd.grad(
        L.sum(), [theta, theta_dot], create_graph=True
    ) # [N,1]

    # Jacobian = grad ( grad_theta_dot ) wrt theta
    Jacobian, Hessian = torch.autograd.grad(
        grad_theta_dot.sum(), [theta, theta_dot], create_graph=True
    ) # [N ,1]
    # Jacobian should be a matrix however since my function L is a scalar
    # I can directly use the vectors always I think for now...

    theta_ddot = (grad_theta - Jacobian * theta_dot)/(Hessian + 1e-8)
    return (theta_dot, theta_ddot)

class BaselineNN(nn.Module):
    def __init__(self, hidden_layers = (32, 64, 128, 128, 64, 32), activation_fn = nn.ReLU()):
        super().__init__()
        self.net = make_mlp(input_dim=2,
                            hidden_layers=hidden_layers,
                            output_dim=2,
                            activation_fn=activation_fn)

    def forward(self, theta, theta_dot):
        state_nn = torch.cat([theta, theta_dot], dim=-1)
        return self.net(state_nn).squeeze(-1)