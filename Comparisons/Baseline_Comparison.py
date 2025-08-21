import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Model
import Training
import Simulation
import random

# Fixed Parameters for LNN

layer_struct = (32, 64, 256, 128, 64)
states_initial = np.array([[random.uniform(-np.pi/2, np.pi/2), random.uniform(-2, 2)] for _ in range(5)])
length = 1.0
batch_size = 256
time_max = 30
dt = 0.01
epochs = 50
activation_function = nn.Softmax(dim=1)

# Creating True Data

# Choose a new initial state for fair evaluation
test_initial_state = np.array([random.uniform(-np.pi/2, np.pi/2), random.uniform(-2, 2)])  # Different from training init

theta_arr, theta_dot_arr, time_arr = Simulation.Solver(
    Simulation.Euler_Lagrange_Equation,
    test_initial_state[0], test_initial_state[1],
    time_max, dt, length
)

test_states = np.stack([theta_arr, theta_dot_arr], axis=-1)  # [T, 2]

# Testing Model

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")


theta_start = torch.tensor([[test_states[0, 0]]], device=device, dtype=torch.float32)      # shape [1, 1]
theta_dot_start = torch.tensor([[test_states[0, 1]]], device=device, dtype=torch.float32)  # shape [1, 1]

## Training LNN

lnn_predictions = [[float(theta_start.item()), float(theta_dot_start.item())]]
nn_predictions = [[float(theta_start.item()), float(theta_dot_start.item())]]

# Training Models based on different Window Sizes
LNN_Model = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_Model, 
    states_initial=states_initial, 
    length = length, 
    t_max=time_max, 
    dt=dt, 
    window_size = 10,
    batch_size=batch_size, 
    epochs=epochs, 
    resume = False)

LNN_Model = LNN_Model.to(device)
LNN_Model.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_Model, theta, theta_dot, dt)
    theta = theta.detach()
    theta_dot = theta_dot.detach()
    lnn_predictions.append([float(theta.item()), float(theta_dot.item())])

NN_Model = Model.BaselineNN(hidden_layers=(32, 64, 128, 256, 512, 256, 128, 64, 32),activation_fn=nn.ReLU())
Training.BaselineNN_Train(
    nn_model=NN_Model,
    states_initial=states_initial,
    length = length,
    t_max=time_max,
    dt=dt,
    window_size = 1,
    batch_size = batch_size,
    epochs = 2*epochs,
)

NN_Model = NN_Model.to(device)
NN_Model.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    out = NN_Model(theta, theta_dot)
    theta = out[:, 0].unsqueeze(-1).detach()
    theta_dot = out[:, 1].unsqueeze(-1).detach()
    nn_predictions.append([float(theta.item()), float(theta_dot.item())])

LNN_pred = np.array(lnn_predictions)
NN_pred = np.array(nn_predictions)

LNN_Energy = (LNN_pred[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(LNN_pred[:,0]))
NN_Energy = (NN_pred[:, 1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(NN_pred[:,0]))

print(LNN_pred.shape, test_states.shape)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_arr, LNN_pred[:, 0], label="LNN")
plt.plot(time_arr, NN_pred[:, 0], label="Baseline")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.title("LNN vs Baseline ( Theta )")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_arr, LNN_Energy, label="LNN")
plt.plot(time_arr, NN_Energy, label="Baseline")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("LNN vs Baseline ( Energy )")
plt.grid(True)

plt.tight_layout()
plt.show()

