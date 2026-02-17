# Neuro-Fuzzy Computing Coursework

This repository contains solutions and implementations for the **Neuro-Fuzzy Computing** course (Fall 2025-2026). The project focuses on the mathematical foundations of neural networks, advanced optimization algorithms, and fuzzy logic systems.

It includes theoretical proofs derived in LaTeX and practical implementations in **MATLAB** and **Python**, featuring visualizations of optimization landscapes, decision boundaries, and network performance.

## 📂 Project Structure & Key Implementations

### [Problem Set 1: Neural Network Fundamentals](./Problem_Set_1)
* **Optimization Landscapes (Ex 1):** Analysis and contour plotting of the function $f(x,y)=x^4+y^4-4xy+1$, identifying global minima and saddle points.
* **Activation Function Analysis (Ex 4, 5):**
    * Comparative study of **Log-Sigmoid vs. ReLU** networks.
    * Implementation of **Swish** activation with **Leaky ReLU** output layers.
* **ADALINE & LMS Algorithm (Ex 6, 8):**
    * Multi-class classification using ADALINE.
    * Visualization of the **MSE performance surface** and the **LMS weight trajectory** converging to the optimal solution $w^*$.
* **Backpropagation (Ex 9):** Full matrix-based implementation of an MLP to approximate non-linear functions (e.g., g(p) = 1 + sin(pπ/3)), including hyperparameter tuning ($S^1$ size, learning rate).

### [Problem Set 2: Advanced Optimization & CNNs](./Problem_Set_2)
* **Conjugate Gradient (Ex 3):** Implementation of the Conjugate Gradient method for quadratic minimization.
* **Modern Optimizers (Ex 5):** Implementation of the **Adadelta** optimizer.
    * Visualization of trajectories on standard vs. rotated loss surfaces.
    * Analysis of convergence behavior with different learning rates.
* **Efficient Convolution (Ex 9):**
    * Implementation of 2D convolution using **Toeplitz Matrices** (converting convolution to matrix multiplication).
    * Performance benchmarking: Direct Convolution vs. Matrix Multiplication.

### [Problem Set 3: Recurrent Neural Networks](./Problem_Set_3)
* **Time Series Prediction:**
    * Generation of synthetic data using an **Auto-Regressive (AR)** model.
    * Implementation of **RNNs (GRU/LSTM)** to predict temporal sequences.
    * Analysis of training stability and Mean Squared Error (MSE) convergence.

## 🛠️ Technologies
* **MATLAB**: Used for Optimization (Adadelta, Conjugate Gradient), ADALINE, and MLP Backpropagation.
* **Python**: Used for RNN/LSTM time-series modeling and Deep Learning tasks.

## 📊 Visualizations

Below are selected visualizations generated from the code in this repository:

| LMS Optimization Trajectory | Adadelta on Rotated Surface |
|:---------------------------:|:---------------------------:|
| <img src="./Problem_Set_1/figures/lms_trajectory.png" width="400" alt="LMS Trajectory Ex 8"> | <img src="./Problem_Set_2/figures/adadelta_rotated.png" width="400" alt="Adadelta Ex 5"> |
| *Visualizing the path of weights on the MSE Error Surface* | *Adadelta optimizer navigating a correlated loss landscape* |

| MLP Function Approximation | Convolution vs Matrix Mult |
|:--------------------------:|:--------------------------:|
| <img src="./Problem_Set_1/figures/function_approx.png" width="400" alt="Backprop Ex 9"> | <img src="./Problem_Set_2/figures/conv_benchmark.png" width="400" alt="Convolution Ex 9"> |
| *Approximating $1+\sin(p\pi/3)$ with variable hidden units* | *Benchmarking Toeplitz matrix multiplication speed* |

## 📜 License
This project is for educational purposes. Please attribute if using the code.
