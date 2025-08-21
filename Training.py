import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import Simulation as sim
import Model as Model


def LNN_Workout(lnn_model, states_initial, length, t_max, dt, window_size, epochs, batch_size,
            save_name = "save_latest.pth", resume = True):

    directory = Path("saves")
    directory.mkdir(parents=True, exist_ok=True)
    save_path = directory / save_name

    if resume and save_path.exists():
        try:
            checkpoint = torch.load(save_path)
            lnn_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Could not load checkpoint {save_path}: {e}")
            print("Starting from the go...")
    else:
        if resume:
            print(f"No checkpoint found  at {save_path}, Starting Training from the go... \n 13 come on... 14 come on... Get it 17... Get it 18... ")
        else:
            print(f"Starting Training. \n 7.. 8.. 9.. 10.. A Machine here")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    lnn_model = lnn_model.to(device)


    lnn_model.train()

    # Simulation data
    theta_total = np.array([])
    theta_dot_total = np.array([])
    for state_initial in states_initial:
        theta_init = state_initial[0]
        theta_dot_init = state_initial[1]
        theta_arr , theta_dot_arr, time_arr = sim.Solver(
            sim.Euler_Lagrange_Equation,
            theta_init, theta_dot_init,
            t_max, dt,length
        )
        theta_total = np.append(theta_total, theta_arr)
        theta_dot_total = np.append(theta_dot_total, theta_dot_arr)

    state_arr = np.stack([theta_total, theta_dot_total], axis = -1)

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
    epoch = 0
    avg_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs_batch, targets_batch in loader:
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad()
            theta_batch = inputs_batch[:, 0].unsqueeze(-1)
            theta_dot_batch = inputs_batch[:, 1].unsqueeze(-1)

            for i in range(window_size):
                theta_dot_batch, theta_ddot_batch = Model.LNN_Euler_Lagrange(lnn_model, theta_batch, theta_dot_batch)
                theta_dot_batch = theta_dot_batch + theta_ddot_batch * dt
                theta_batch = theta_batch + theta_dot_batch*dt

            prediction_batch = torch.cat([theta_batch, theta_dot_batch], dim = -1)
            loss = loss_fn(prediction_batch, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs_batch)
        avg_loss = epoch_loss / len(dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.8f}")

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": lnn_model.state_dict(),
        "loss": avg_loss
    }, save_path)
    print(f"Saved final checkpoint to {save_path} (loss={avg_loss:.8f})")

def BaselineNN_Train(nn_model, states_initial, length, t_max, dt, window_size, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    nn_model = nn_model.to(device)

    nn_model.train()

    theta_total = np.array([])
    theta_dot_total = np.array([])
    for state_initial in states_initial:
        theta_init = state_initial[0]
        theta_dot_init = state_initial[1]
        theta_arr, theta_dot_arr, time_arr = sim.Solver(
            sim.Euler_Lagrange_Equation,
            theta_init, theta_dot_init,
            t_max, dt, length
        )
        theta_total = np.append(theta_total, theta_arr)
        theta_dot_total = np.append(theta_dot_total, theta_dot_arr)

    state_arr = np.stack([theta_total, theta_dot_total], axis=-1)

    # Target Generation

    inputs = state_arr[:-window_size]
    targets = state_arr[window_size:]

    # Create DataLoader
    tensor_inputs = torch.tensor(inputs, dtype=torch.float32)
    tensor_targets = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(tensor_inputs, tensor_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and Loss
    optimizer = optim.Adam(nn_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs_batch, targets_batch in loader:
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)

            optimizer.zero_grad()
            theta_batch = inputs_batch[:, 0].unsqueeze(-1)
            theta_dot_batch = inputs_batch[:, 1].unsqueeze(-1)

            for i in range(window_size):
                out = nn_model(theta_batch, theta_dot_batch)  # [Batch , 2]
                theta_batch = out[:, 0].unsqueeze(-1)  # [Batch, 1]
                theta_dot_batch = out[:, 1].unsqueeze(-1)  # [Batch, 1]

            prediction_batch = torch.cat([theta_batch, theta_dot_batch], dim = -1)
            loss = loss_fn(prediction_batch, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs_batch)
        avg_loss = epoch_loss / len(dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.8f}")




