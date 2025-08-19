import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import Simulation as sim
import Model as Model

def trainer(lnn_model, state_initial, length, t_max, dt, window_size, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    lnn_model = lnn_model.to(device)
    lnn_model.train()

    # Simulation data
    theta_init = state_initial[0]
    theta_dot_init = state_initial[1]
    theta_arr , theta_dot_arr, time_arr = sim.Solver(
        sim.Euler_Lagrange_Equation,
        theta_init, theta_dot_init,
        t_max, dt,length
    )

    state_arr = np.stack([theta_arr, theta_dot_arr], axis = -1)

    # Target Generation

    inputs = state_arr[:-window_size]
    targets = state_arr[window_size:]

    # Create DataLoader
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    tensor_targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(tensor_inputs, tensor_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and Loss
    optimizer = optim.Adam(lnn_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs_batch, targets_batch in loader:
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad()
            theta_batch = inputs_batch[:, 0]
            theta_dot_batch = inputs_batch[:, 1]

            for i in range(window_size):
                theta_dot_batch, theta_ddot_batch = Model.LNN_Euler_Lagrange(lnn_model, theta_batch, theta_dot_batch)
                theta_dot_batch = theta_dot_batch + theta_ddot_batch * dt
                theta_batch = theta_batch + theta_dot_batch*dt

            prediction_batch = torch.stack([theta_batch, theta_dot_batch], dim = -1)
            loss = loss_fn(prediction_batch, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs_batch)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.5f}")

    print("Training complete.")


# state_initial = np.array([np.pi/6, 0])
# length = 1.0
# window_size = 2
# batch_size = 100
# # Simulation data
# theta_init = state_initial[0]
# theta_dot_init = state_initial[1]
# theta_arr, theta_dot_arr, time_arr = sim.Solver(
#     sim.Euler_Lagrange_Equation,
#     theta_init, theta_dot_init,
#     10, 0.01, length
# )
#
# state_arr = np.stack([theta_arr, theta_dot_arr], axis = -1)
#
# # Target Generation
#
# inputs = state_arr[:-window_size]
# targets = state_arr[window_size:]
#
# tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
# tensor_targets = torch.tensor(targets, dtype=torch.float32)
# dataset = TensorDataset(tensor_inputs, tensor_targets)
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# for inputs_batch, targets_batch in loader:
#     theta_batch = inputs_batch[:, 0]
#     print(len(inputs_batch))

#
# print("State Array: ", state_arr)
# print("Inputs Array: ",inputs)
# print("Target Array: ",targets)