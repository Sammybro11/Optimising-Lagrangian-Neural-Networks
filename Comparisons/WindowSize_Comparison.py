import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import Model
import Training
import Simulation
import random

# Fixed Parameters for Comparison
layer_struct = (32, 64, 256, 128, 64)
states_initial = np.array([[random.uniform(-np.pi/2, np.pi/2), random.uniform(-2, 2)] for _ in range(10)])
length = 1.0
batch_size = 512
time_max = 10
dt = 0.01
epochs = 50
activation_function = nn.Softmax(dim=1)

# Creating True Data

# Choose a new initial state for fair evaluation
test_initial_state = np.array([random.uniform(-np.pi/2, np.pi/2), random.uniform(-2.5, 2.5)])  # Different from training init

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

window_1_pred = [[float(theta_start.item()), float(theta_dot_start.item())]]
window_3_pred = [[float(theta_start.item()), float(theta_dot_start.item())]]
window_5_pred = [[float(theta_start.item()), float(theta_dot_start.item())]]
window_10_pred = [[float(theta_start.item()), float(theta_dot_start.item())]]

# Training Models based on different Window Sizes
LNN_window_1 = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_window_1, states_initial=states_initial, length = length, t_max=time_max, dt=dt, window_size = 1, batch_size=batch_size, epochs=epochs, resume = False)

LNN_window_1 = LNN_window_1.to(device)
LNN_window_1.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_window_1, theta, theta_dot, dt)
    theta = theta.detach()
    theta_dot = theta_dot.detach()
    window_1_pred.append([float(theta.item()), float(theta_dot.item())])


LNN_window_3 = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_window_3, states_initial=states_initial, length = length, t_max=time_max, dt=dt, window_size = 3, batch_size=batch_size, epochs=epochs, resume = False)

LNN_window_3 = LNN_window_3.to(device)
LNN_window_3.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_window_3, theta, theta_dot, dt)
    theta = theta.detach()
    theta_dot = theta_dot.detach()
    window_3_pred.append([float(theta.item()), float(theta_dot.item())])


LNN_window_5 = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_window_5, states_initial=states_initial, length = length, t_max=time_max, dt=dt, window_size = 5, batch_size=batch_size, epochs=epochs, resume = False)

LNN_window_5 = LNN_window_5.to(device)
LNN_window_5.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_window_5, theta, theta_dot, dt)
    theta = theta.detach()
    theta_dot = theta_dot.detach()
    window_5_pred.append([float(theta.item()), float(theta_dot.item())])


LNN_window_10 = Model.LNN(hidden_layers=layer_struct,activation_fn=activation_function)
Training.LNN_Workout(
    lnn_model=LNN_window_10, states_initial=states_initial, length = length, t_max=time_max, dt=dt, window_size = 10, batch_size=batch_size, epochs=epochs, resume = False)

LNN_window_10 = LNN_window_10.to(device)
LNN_window_10.eval()
theta = theta_start
theta_dot = theta_dot_start
for i in range(len(test_states) - 1):
    theta, theta_dot = Simulation.rk4_step_LNN(Model.LNN_Euler_Lagrange, LNN_window_10, theta, theta_dot, dt)
    theta = theta.detach()
    theta_dot = theta_dot.detach()
    window_10_pred.append([float(theta.item()), float(theta_dot.item())])


window_1_states = np.array(window_1_pred)
window_3_states = np.array(window_3_pred)
window_5_states = np.array(window_5_pred)
window_10_states = np.array(window_10_pred)

true_states = test_states

true_energy = (true_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(true_states[:,0]))
window_1_energy = (window_1_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(window_1_states[:,0]))
window_3_energy = (window_3_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(window_3_states[:,0]))
window_5_energy = (window_5_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(window_5_states[:,0]))
window_10_energy = (window_10_states[:,1]**2)*(length**2)/2 + (9.8066 * length) * ( 1 - np.cos(window_10_states[:,0]))

plt.figure(figsize=(18, 12))

plt.subplot(4, 2, 1)
plt.plot(time_arr, true_states[:, 0], label='True θ')
plt.plot(time_arr, window_1_states[:, 0], '--',c = "green", label='Window 1')
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 2)
plt.plot(time_arr, true_energy, label='True θ')
plt.plot(time_arr, window_1_energy, '-.',c = "green", label='Window 1')
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 3)
plt.plot(time_arr, true_states[:, 0], label='True θ')
plt.plot(time_arr, window_3_states[:, 0], '--',c = "red",  label='Window 3')
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 4)
plt.plot(time_arr, true_energy, label='True θ')
plt.plot(time_arr, window_3_energy, '-.',c = "red", label='Window 3')
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 5)
plt.plot(time_arr, true_states[:, 0], label='True θ')
plt.plot(time_arr, window_5_states[:, 0], '--',c = "purple",  label='Window 5')
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 6)
plt.plot(time_arr, true_energy, label='True θ')
plt.plot(time_arr, window_5_energy, '-.',c = "purple", label='Window 5')
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 7)
plt.plot(time_arr, true_states[:, 0], label='True θ')
plt.plot(time_arr, window_10_states[:, 0], '--',c = "black", label='Window 10')
plt.xlabel("Time")
plt.ylabel("Theta (θ)")
plt.grid(True)
plt.legend()
plt.subplot(4, 2, 8)
plt.plot(time_arr, true_energy, label='True θ')
plt.plot(time_arr, window_10_energy, '-.',c = "purple", label='Window 10')
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

