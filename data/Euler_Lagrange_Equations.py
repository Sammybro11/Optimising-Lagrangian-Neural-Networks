import numpy as np
import os
from src.utils import rk4_step

GRAVITY = 9.8066

def Solver(equation, state_init, t_max, dt):
    """
    Generates time series data on states for a given Euler Lagrange equation
    :param equation: Euler Lagrange equation outputs [velocity, acceleration]
    :param state_init: Initial state [displacement, velocity]
    :param t_max: Maximum time for Time series
    :param dt: Time step
    :return: Time series data series [number of steps, state_length]
    """
    number_of_steps = int(t_max/dt)
    t_eval = np.linspace(start = 0, stop = t_max, num = number_of_steps)
    state = state_init.copy()

    state_history = np.zeros((number_of_steps, state.shape[0]))
    state_history[0] = state

    for i in range(1, number_of_steps):
        state = rk4_step(equation, state, dt)
        state_history[i] = state

    return state_history


def Create_SP(length):
    """
    Creates Euler Lagrange equation for a Simple Pendulum with given length
    :param length:
    :return: EL Equation of Simple Pendulum
    """
    def Simple_Pendulum(state):
        """
        Euler Lagrange equation for a Simple Pendulum
        :param state: State variable of the system [theta, angular velocity]
        :return: [theta_dot, theta_ddot]: Angular Velocity and Angular Acceleration of the Pendulum
        """
        theta_ddot =  - (GRAVITY * np.sin(state[0]))/length
        return np.array([state[1], theta_ddot])
    return Simple_Pendulum

def Create_DP(length_1, length_2, mass_1, mass_2):

    def Double_Pendulum(state):
        """
        Euler Lagrange equation for a Double Pendulum
        :param state: State variable of the system [theta_1, theta_2, omega_1, omega_2]
        :return: state_derv: Derivative variable [omega_1, omega_2, alpha_1, alpha_2]
        """
        theta_1, theta_2, omega_1, omega_2 = state

        delta = theta_1 - theta_2
        denominator = (2*mass_1 + mass_2 - mass_2 * np.cos(2*delta))

        alpha_1 = (
        -GRAVITY * (2*mass_1 + mass_2) * np.sin(theta_1)
        - mass_2 * GRAVITY * np.sin(theta_1 - 2*theta_2)
        - 2*np.sin(delta) * mass_2 * (omega_2**2 * length_2 + omega_1**2 * length_1 * np.cos(delta))
        ) / (length_1 * denominator)

        alpha_2 = (
        2 * np.sin(delta)
        * (omega_1**2 * length_1 * (mass_1 + mass_2)
           + GRAVITY * (mass_1 + mass_2) * np.cos(theta_1)
           + omega_2**2 * length_2 * mass_2 * np.cos(delta))
        ) / (length_2 * denominator)

        return np.array([omega_1, omega_2, alpha_1, alpha_2])

    return Double_Pendulum

# def Create_3BP(masses):
#     m1, m2, m3 = masses
#
#     def Three_Body_Problem(state):
#         x_1, y_1, x_2, y_2, x_3, y_3 = state[:6]
#         vx_1, vy_1, vx_2, vy_2, vx_3, vy_3 = state[6:]
#
#         r_12 = np.array([x_2 - x_1, y_2-y_1])
#         return 0
#     return Three_Body_Problem

def Simulate(system_name, equation, state_init, t_max, dt, save_dir = "sims/"):

    state_history = Solver(equation, state_init, t_max, dt)

    os.makedirs(save_dir, exist_ok = True)

    init_str = "_".join([f"{x:.2f}" for x in state_init])
    filename = f"{system_name}_init_{init_str}.npy"
    filepath = os.path.join(save_dir, filename)

    np.save(filepath, state_history)

    print(f"Saved initial state history to {filepath}")
    return filepath