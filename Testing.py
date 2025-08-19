#%%
import Model
import Simulation
import Training

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
# Initialization

state_initial = np.array([np.pi/10, 3])
length = 1.0
window = 10
batch_size = 32
time_max = 50
dt = 0.01
epochs = 50

# Training Model
LNN_1 = Model.LNN()
LNN_1.load_state_dict(torch.load("saves/save01.pth"))

Training.trainer(LNN_1, state_initial, length, time_max, dt, window, epochs, batch_size)

# Measuring True Output on new initial state

# Choose a new initial state for fair evaluation
test_initial_state = np.array([np.pi/8, 2])  # Different from training init

theta_arr, theta_dot_arr, time_arr = Simulation.Solver(
    Simulation.Euler_Lagrange_Equation,
    test_initial_state[0], test_initial_state[1],
    time_max, dt, length
)

test_states = np.stack([theta_arr, theta_dot_arr], axis=-1)  # [T, 2]
# Saving Model
# 1. Create models directory
MODEL_PATH = Path("saves")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = f"save01.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=LNN_1.state_dict(),  # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

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
    theta_dot_out, theta_ddot = Model.LNN_Euler_Lagrange(LNN_1, theta, theta_dot)  # shapes [1, 1] each
    theta_dot = theta_dot + theta_ddot * dt    # [1, 1]
    theta = theta + theta_dot * dt             # [1, 1]

    pred_states.append([theta.item(), theta_dot.item()])

pred_states = np.array(pred_states)   # [T-1, 2]

# ---- Compute Test Loss ----

true_states = test_states[1:len(pred_states)+1]  # Align shapes
mse = np.mean((pred_states - true_states) ** 2)
print(f"Test MSE loss: {mse:.6f}")

# ---- Plot Results ----

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(time_arr[1:len(pred_states)+1], true_states[:, 0], label='True θ')
plt.plot(time_arr[1:len(pred_states)+1], pred_states[:, 0], '--', label='Predicted θ')
plt.xlabel("Time")
plt.ylabel("Theta")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_arr[1:len(pred_states)+1], true_states[:, 1], label='True θ_dot')
plt.plot(time_arr[1:len(pred_states)+1], pred_states[:, 1], '--', label='Predicted θ_dot')
plt.xlabel("Time")
plt.ylabel("Theta_dot")
plt.legend()

plt.suptitle("LNN Model Test Result")
plt.tight_layout()
plt.show()