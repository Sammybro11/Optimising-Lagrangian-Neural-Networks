import torch
import torch.nn as nn
import numpy as np

class LNN(nn.Module):
    def __init__(self):
        super(LNN, self).__init__()
        self.Linear_Stack = nn.Sequential(
            nn.Linear(2, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
        )

    def forward(self, theta, theta_dot):
        state = torch.cat([theta, theta_dot], dim=-1) # I want a [N,1] for each theta and theta_dot
        return self.Linear_Stack(state).squeeze(-1) # This will give out a [N]

def LNN_Euler_Lagrange(lnn, theta, theta_dot):
    theta.requires_grad = True
    theta_dot.requires_grad = True
    L = lnn(theta, theta_dot) # here you get that [N]

    grad_theta, grad_theta_dot = torch.autograd.grad(
        L.sum(), [theta, theta_dot], create_graph=True
    )

    # Jacobian = grad ( grad_theta_dot ) wrt theta
    Jacobian, Hessian = torch.autograd.grad(
        grad_theta_dot.sum(), [theta, theta_dot], create_graph=True
    )
    # Jacobian should be a matrix however since my function L is a scalar
    # I can directly use the vectors always I think for now...

    theta_ddot = (grad_theta - Jacobian * theta_dot)/Hessian
    return (theta_dot, theta_ddot)