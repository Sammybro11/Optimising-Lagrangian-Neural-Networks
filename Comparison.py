import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Model
import Training
import Simulation
import random
from tqdm import tqdm

# Fixed Parameters for LNN
torch.manual_seed(0)
random.seed(0)

layer_struct = (64, 256, 256, 64)
states_initial = np.array([[random.uniform(-np.pi/3, np.pi/3), random.uniform(-1, 1)] for _ in range(5)])
length = 1.0
batch_size = 512
time_max = 20
dt = 0.01
epochs = 50
activation_function = nn.Softplus()

# Creating True Data

# Choosing new initial values for testing
test_initial_state = np.array([random.uniform(-np.pi/3, np.pi/3), random.uniform(-1, 1)])  # Different from training init

theta_arr, theta_dot_arr, time_arr = Simulation.Solver(
    Simulation.Euler_Lagrange_Equation,
    test_initial_state[0], test_initial_state[1],
    time_max/2, dt, length
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

LNN_Model = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_Model, 
    states_initial=states_initial, 
    length = length, 
    t_max=time_max, 
    dt=dt, 
    window_size = 5,
    batch_size=batch_size, 
    epochs=epochs, 
    resume = False)

LNN_Model = LNN_Model.to(device)
LNN_Model.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in tqdm(range(len(test_states) - 1), desc = 'Testing LNN Model: '):
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
True_Energy = (test_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(test_states[:,0]))

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

axs[0,0].plot(time_arr, LNN_pred[:, 0], label="LNN")
axs[0,0].plot(time_arr, test_states[:, 0], label="True")
axs[0,0].legend()
axs[0,0].set_xlabel("Time")
axs[0,0].set_ylabel("Theta (θ)")
axs[0,0].set_title("LNN (Theta)")
axs[0,0].grid(True)

axs[0,1].plot(time_arr, NN_pred[:, 0], label="NN")
axs[0,1].plot(time_arr, test_states[:, 0], label="True")
axs[0,1].legend()
axs[0,1].set_xlabel("Time")
axs[0,1].set_ylabel("Theta (θ)")
axs[0,1].set_title("Neural Network (Theta)")
axs[0,1].grid(True)

axs[1,0].plot(time_arr, LNN_Energy, label="LNN")
axs[1,0].plot(time_arr, True_Energy, label="True")
axs[1,0].legend()
axs[1,0].set_xlabel("Time")
axs[1,0].set_ylabel("Energy")
axs[1,0].set_title("LNN (Energy)")
axs[1,0].grid(True)

axs[1,1].plot(time_arr, NN_Energy, label="NN")
axs[1,1].plot(time_arr, True_Energy, label="True")
axs[1,1].legend()
axs[1,1].set_xlabel("Time")
axs[1,1].set_ylabel("Energy")
axs[1,1].set_title("Neural Network (Energy)")
axs[1,1].grid(True)

axs[0,1].set_xlim(axs[0,0].get_xlim())
axs[0,1].set_ylim(axs[0,0].get_ylim())

axs[1,1].set_xlim(axs[1,0].get_xlim())
axs[1,1].set_ylim(axs[1,0].get_ylim())

plt.tight_layout()
plt.show()


