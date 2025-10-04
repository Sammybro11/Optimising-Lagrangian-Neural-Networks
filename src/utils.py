import numpy as np

def rk4_step(function, state, dt):
    """
    RK4 step function for time independent equations
    :param function: Euler-Lagrange function outputs [velocity, acceleration]
    :param state: State variable of the system [displacement, velocity]
    :param dt: time step
    :return: state_next: Next time step State variable of the system [displacement, velocity]
    """
    k1 = function(state[0], state[1])
    k2 = function(state[0] + 0.5 * dt * k1[0], state[1] + 0.5 * dt * k1[1])
    k3 = function(state[0] + 0.5 * dt * k2[0], state[1] + 0.5 * dt * k2[1])
    k4 = function(state[0] + dt * k3[0], state[1] + dt * k3[1])
    state_next = state + (dt/6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state_next