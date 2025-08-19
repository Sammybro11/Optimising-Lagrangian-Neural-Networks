import numpy as np
import matplotlib.pyplot as plt
debug = False

def Euler_Lagrange_Equation(theta, theta_dot, length):
    theta_ddot =  - (9.8066 * np.sin(theta))/length
    return np.array([theta_dot, theta_ddot])

def rk4_step(function, state, time, dt, *args):
    k1 = function(state[0], state[1], *args)
    k2 = function(state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1], *args)
    k3 = function(state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1], *args)
    k4 = function(state[0] + dt * k3[0], state[1] + dt * k3[1], *args)
    state_next = state + (dt/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state_next

def rk4_step_LNN(function, lnn, theta, theta_dot, dt):
    k1 = function(lnn, theta, theta_dot)
    k2 = function(lnn, theta + 0.5 * dt * k1[0], theta_dot + 0.5 * dt * k1[1])
    k3 = function(lnn, theta + 0.5 * dt * k2[0], theta_dot + 0.5 * dt * k2[1])
    k4 = function(lnn, theta + dt * k3[0], theta_dot + dt * k3[1])
    theta_next = theta + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    theta_dot_next = theta_dot + (dt/6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return (theta_next, theta_dot_next)

def Solver(Equation, theta_initial, theta_dot_initial, t_max, dt, *args):
    number_of_steps = int(t_max/dt)
    t_eval = np.linspace(0, t_max, number_of_steps)

    state = np.array([theta_initial, theta_dot_initial])
    theta_history = np.zeros(number_of_steps)
    theta_dot_history = np.zeros(number_of_steps)

    theta_history[0] = state[0]
    theta_dot_history[0] = state[1]

    for i in range(1, number_of_steps):
        state = rk4_step(Equation, state, t_eval, dt, *args)
        theta_history[i] = state[0]
        theta_dot_history[i] = state[1]

    return theta_history, theta_dot_history, t_eval

if debug == True:
    theta_history, theta_dot_history, t_eval = Solver(
        Euler_Lagrange_Equation,
        np.pi / 6,  # 30 degrees initial angle
        6.05,          # Initial velocity
        10,         # Max time
        0.01,       # Time step
        1.0         # Pendulum length (passed as *args)
    )

    # --- Plotting the Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, theta_history, label='Theta (Angle)')
    plt.title('Simple Pendulum Simulation (RK4)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radians)')
    plt.grid(True)
    plt.legend()
    plt.show()
