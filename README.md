# Optimising Lagrangian Neural Networks

This project is a demonstration of Lagrangian Neural Networks (LNNs) superiority to Standard MLP
due the  utilisation of Euler-Lagrange Equation which allows the LNN to follow constraints of Energy and Lagrangian Mechanics. 
To make the analysis and comparison fair, the LNNs and Standard MLPs are optimised for their architecture stack and then compared.

Different kinds of systems which follow Lagrangian Dynamics are considered for comparison this includes

- Simple Pendulums
- Compound Pendulums
- 

## Project Structure
```text
Optimising Lagrangian Neural Networks
|
|--- src
|    |--- __init.py__
|    |--- Simulation.py
|    |--- Model.py
|    |--- Training.py
|--- main.py
|--- requirements.txt
|--- README.md
|--- LICENSE
```

### Cloning Instructions
**Installation**
```
git clone https://github.com/Sammybro11/Optimising-Lagrangian-Neural-Networks
cd Optimising-Lagrangian-Neural-Networks
```
**Python Virtual Environment**
```
python3 -m venv env
source env/bin/activate
```
**Installing required Packages**
```
pip install -r requirements.txt
```
---
## Theory
Let us consider the case of a simple Pendulum

The **Lagrangian** $`L(\phi, \dot{\phi}, t)`$ describes the motion of the pendulum, where:
- $`\phi `$ = Vertical angle of Pendulum from support
- $`\dot{\phi}`$ = Angular velocity of Pendulum from support
- $`t`$ = Time variable

The Euler-Lagrange equation is:
```math
\frac{d}{dt} \nabla_{\dot{\phi}} L = \nabla_\phi L
```
where:
- $`\nabla_\phi L`$ is the gradient w.r.t. $`\phi`$
- $`\nabla_{\dot{\phi}} L`$ is the gradient w.r.t. $`\dot{\phi}`$

We want to solve for $`\ddot{\phi}`$:
```math
\frac{d}{dt} \nabla_{\dot{\phi}} L = \nabla_\phi L
```
By applying Chain Rule
```math

\frac{\partial}{\partial t} \nabla_{\dot{\phi}} L + \nabla^2_{\dot{\phi} \dot{\phi}} L \, \ddot{\phi} + \nabla^2_{\phi \dot{\phi}} L \, \dot{\phi} = \nabla_\phi L
```
```math
\nabla^2_{\dot{\phi} \dot{\phi}} L \, \ddot{\phi} = \nabla_\phi L - \nabla^2_{\phi \dot{\phi}} L \, \dot{\phi} - \frac{\partial}{\partial t} \nabla_{\dot{\phi}} L
```
Therefore,
```math
\ddot{\phi} = [\nabla^2_{\dot{\phi} \dot{\phi}} L]^{-1} [\nabla_\phi L - \nabla^2_{\phi \dot{\phi}} L \, \dot{\phi} - \frac{\partial}{\partial t} \nabla_{\dot{\phi}} L]
```
---

## Model Structure

The second time derivative obtained from the LNN + Euler Lagrange Equation ( `src/Model.py` )
is used to obtain the next time step values of the state variables of the 
system using RK4 Method ( `src/Simulation.py` ).
This is then used to obtain Mean Squared Error Loss used in backpropagation using Adam Optimizer.

![Model_Struct](https://www.researchgate.net/publication/355142323/figure/fig1/AS:1078319500013570@1634102811778/Momentum-conserving-Lagrangian-neural-network.png)

## Analysis of Simple Pendulum

## Analysis of Pendulum on Moving Support

## Analysis of Duffing Oscillator






## 📜 License
Apache License 2.0. Free to use and modify.
