# Lagrangian Neural Networks for Pendulum Dynamics

This repository implements **Lagrangian Neural Networks (LNNs)** and provides a framework for their **training, optimization, and analysis** on simple pendulum systems. The goal is to leverage physics-informed machine learning to better capture the dynamics of mechanical systems compared to traditional black-box neural networks.

---

## ⚙️ Installation

Clone this repository and install dependencies:

git clone <repo-url>
cd <repo>
pip install -r requirements.txt

---

## 📂 Project Structure

.
├── Model.py              # Defines neural network and LNN architectures
├── Training.py           # Training pipeline for standard NN and LNN models
├── Testing.py            # Evaluation of trained models on pendulum dynamics
├── Simulation.py         # Simulations using trained models on pendulum trajectories
├── Model_Comparison.ipynb# Notebook for experiments, plots, and analysis

---

## 🚀 Concepts Used

### 1. Lagrangian Mechanics
- Classical mechanics describes systems using the **Lagrangian**:  
  L(q, q̇) = T(q̇) − V(q)  
  where `T` is kinetic energy and `V` is potential energy.  
- The **Euler–Lagrange equations** yield the equations of motion:  
  d/dt (∂L/∂q̇) − ∂L/∂q = 0.  
- This approach guarantees **energy-consistent dynamics**, unlike unconstrained neural nets.

### 2. Lagrangian Neural Networks (LNNs)
- Instead of predicting accelerations directly, the network **learns the Lagrangian** from data.  
- From the learned Lagrangian, accelerations are derived using the Euler–Lagrange equations.  
- This ensures predictions **respect conservation laws** and exhibit better physical generalization.

### 3. Optimization & Training
- Implemented in `Training.py` using standard ML workflows:
  - Forward pass: compute predicted dynamics.
  - Loss function: MSE between predicted and true accelerations/trajectories.
  - Backward pass: gradients via autograd.
  - Optimizer: Adam/SGD updates model parameters.
- Regularization may be applied to stabilize training.

### 4. Testing & Evaluation
- `Testing.py` runs evaluation of models on test trajectories.  
- Metrics include trajectory error, energy conservation, and phase-space reconstruction quality.  
- Comparisons between standard Neural Networks and LNNs highlight advantages of physics-informed learning.

### 5. Simulation
- `Simulation.py` uses trained models to simulate pendulum dynamics.  
- Can roll out trajectories forward in time and compare with ground-truth ODE solutions.  
- Demonstrates long-term stability of LNN predictions versus standard neural networks.

### 6. Model Comparison
- `Model_Comparison.ipynb` runs experiments comparing:
  - Standard Neural Networks (black-box).  
  - Lagrangian Neural Networks (physics-informed).  
- Provides plots of loss curves, predicted vs. true trajectories, and conserved energy analysis.

---

## 📖 Usage

1. Train a Model  
   python Training.py  
   Trains either a standard NN or an LNN on pendulum trajectory data.  

2. Test a Model  
   python Testing.py  
   Evaluates the trained model and computes error metrics.  

3. Run Simulations  
   python Simulation.py  
   Simulates pendulum dynamics using the trained model and visualizes results.  

4. Compare Models  
   jupyter notebook Model_Comparison.ipynb  
   Provides detailed comparisons of NN vs. LNN performance.  

---

## 📊 Example Workflow

1. Define models in `Model.py` (choose NN or LNN).  
2. Train using `Training.py` on pendulum dataset.  
3. Evaluate trained models with `Testing.py`.  
4. Run simulations to validate dynamical stability.  
5. Use `Model_Comparison.ipynb` to analyze which approach works best.  

---

## 🧠 Key Ideas Reinforced
- Neural networks can approximate dynamical systems, but **physics-informed architectures** (LNNs) perform better.  
- Lagrangian formalism ensures learned dynamics respect conservation laws.  
- LNNs are more robust to extrapolation and long-term simulations.  
- Applications extend beyond pendulums to **robotics, molecular dynamics, and physics simulations**.  

---

## 📝 Future Improvements
- Extend experiments to **double pendulums** and chaotic systems.  
- Add **Hamiltonian Neural Networks (HNNs)** for comparison.  
- Use advanced optimization (e.g., symplectic integrators, physics-informed loss terms).  
- Integrate logging and visualization with TensorBoard or Weights & Biases.  

---

## 📜 License
MIT License. Free to use and modify.
