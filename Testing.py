#%%
import Model
import Simulation
import Training

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
# Initialization
layer_struct = (32, 64, 128, 64)
states_initial = np.array([
    [np.pi/5, 2.5],
    [np.pi/6, -0.5],
    [-np.pi/1.5, 0],
])
length = 1.0
window = 10
batch_size = 256
time_max = 20
dt = 0.01
epochs = 50

# Loading Model
LNN_1 = Model.LNN(layer_struct, nn.Softmax(dim=1))
# Training Model

Training.LNN_Workout(LNN_1, states_initial, length, time_max, dt, window, epochs, batch_size, resume= False)

# Measuring True Output on new initial state

# Choose a new initial state for fair evaluation
test_initial_state = np.array([np.pi/2, 2])  # Different from training init

theta_arr, theta_dot_arr, time_arr = Simulation.Solver(
    Simulation.Euler_Lagrange_Equation,
    test_initial_state[0], test_initial_state[1],
    time_max, dt, length
)

test_states = np.stack([theta_arr, theta_dot_arr], axis=-1)  # [T, 2]

# Testing Model

device = torch.device("cuda" if torch.cuda.is_available()
                     else "mps" if torch.backends.mps.is_available() else "cpu")
LNN_1 = LNN_1.to(device)
LNN_1.eval()

pred_states = []
theta = torch.tensor([[test_states[0, 0]]], device=device, dtype=torch.float32)      # shape [1, 1]
theta_dot = torch.tensor([[test_states[0, 1]]], device=device, dtype=torch.float32)  # shape [1, 1]


for i in range(len(test_states) - 1):
    # Euler-Lagrange update expects [batch, 1] tensors
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_1, theta, theta_dot, dt)
    pred_states.append([theta.item(), theta_dot.item()])

pred_states = np.array(pred_states)   # [T-1, 2]

# Compute Total Loss
print(pred_states.shape, test_states.shape)
true_states = test_states[1:len(pred_states)+1]  # Align shapes
mse = np.mean((pred_states - true_states) ** 2)
print(f"Test MSE loss: {mse:.6f}")

# Plot that stuff

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(time_arr[1:len(pred_states)+1], true_states[:, 0], label='True θ')
plt.plot(time_arr[1:len(pred_states)+1], pred_states[:, 0], '--', label='Predicted θ')
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_arr[1:len(pred_states)+1], true_states[:, 1], label='True omega')
plt.plot(time_arr[1:len(pred_states)+1], pred_states[:, 1], '--', label='Predicted omega')
plt.xlabel("Time")
plt.ylabel("Omega")
plt.grid(True)
plt.legend()

plt.suptitle("LNN Model Test Result")
plt.tight_layout()
plt.show()