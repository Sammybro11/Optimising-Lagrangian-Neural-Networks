import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------
# FIXED MLP ARCHITECTURES
# ---------------------------------------------------------------------
def mlp_small(input_dim=2, output_dim=1, activation=nn.Tanh()):
    """Lightweight network, good for simple systems."""
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        activation,
        nn.Linear(32, 32),
        activation,
        nn.Linear(32, output_dim)
    )

def mlp_medium(input_dim=2, output_dim=1, activation=nn.Tanh()):
    """Balanced mid-sized architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        activation,
        nn.Linear(64, 128),
        activation,
        nn.Linear(128, 64),
        activation,
        nn.Linear(64, output_dim)
    )

def mlp_deep(input_dim=2, output_dim=1, activation=nn.Tanh()):
    """Deeper network for complex systems."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        activation,
        nn.Linear(64, 128),
        activation,
        nn.Linear(128, 256),
        activation,
        nn.Linear(256, 128),
        activation,
        nn.Linear(128, 64),
        activation,
        nn.Linear(64, output_dim)
    )

def mlp_wide(input_dim=2, output_dim=1, activation=nn.Tanh()):
    """Wider layers, fewer depth."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        activation,
        nn.Linear(256, 256),
        activation,
        nn.Linear(256, output_dim)
    )

# Optional: You can also have specialized architectures
def mlp_sine(input_dim=2, output_dim=1, omega_0=30):
    """SIREN-like sinusoidal MLP."""
    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.omega_0 = omega_0
        def forward(self, x):
            return torch.sin(self.omega_0 * self.linear(x))
    return nn.Sequential(
        SineLayer(input_dim, 64, omega_0),
        SineLayer(64, 64, omega_0),
        nn.Linear(64, output_dim)
    )


# Model Type Dictionary
MLP_ARCHS = {
    "small": mlp_small,
    "medium": mlp_medium,
    "deep": mlp_deep,
    "wide": mlp_wide,
    "sine": mlp_sine
}


# ---------------------------------------------------------------------
# LNN
# ---------------------------------------------------------------------
class LNN(nn.Module):
    def __init__(self, arch="medium", activation=nn.Tanh()):
        super().__init__()
        if arch not in MLP_ARCHS:
            raise ValueError(f"Unknown architecture '{arch}', choose from {list(MLP_ARCHS.keys())}")
        self.net = MLP_ARCHS[arch](input_dim=2, output_dim=1, activation=activation)

    def forward(self, theta, theta_dot):
        # I want a [N,1] for each theta and theta_dot
        state = torch.cat([theta, theta_dot], dim=-1)  # [N,2]
        return self.net(state).squeeze(-1)         # This will give out a [N]


class LNN(nn.Module):
    def __init__(self, hidden_layers = (32, 64, 128, 64), activation_fn = nn.Softplus()):
        """
        - hidden_layers: tuple[int], e.g. [32, 64, 128, 128, 64, 32]
        - activation_fn: lambda e.g. nn.Softmax(dim=1)
        """
        super().__init__()
        self.net = make_mlp(input_dim=2,
                            hidden_layers=hidden_layers,
                            output_dim=1,
                            activation_fn=activation_fn)




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