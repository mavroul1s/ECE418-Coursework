# Neuro-Fuzzy Computing Coursework

This repository contains solutions and implementations for the **Neuro-Fuzzy Computing** course (Fall 2025-2026), covering the intersection of neural networks, optimization theory, and fuzzy logic. 

The project involves mathematical derivations, theoretical proofs, and implementation of algorithms from scratch using **MATLAB** and **Python**.

## 📂 Project Structure

### [Problem Set 1: Neural Network Fundamentals](./Problem_Set_1)
Focuses on the mathematical foundations of learning algorithms and basic network architectures.
* [cite_start]**Optimization:** Gradient Descent implementation and analytical calculus[cite: 5920].
* [cite_start]**Activation Functions:** Derivations for Sigmoid, Tanh, Swish, and ReLU[cite: 5923].
* [cite_start]**ADALINE & Widrow-Hoff:** Implementation of the LMS algorithm for pattern recognition (classifying 'T', 'G', 'F' pixel patterns)[cite: 6000].
* [cite_start]**Backpropagation:** Full matrix-based implementation of Backpropagation for MLPs to approximate non-linear functions like $g(p) = 1 + \sin(p\pi/3)$[cite: 6040].
* [cite_start]**Stability Analysis:** Maximum stable learning rates and eigenvalues of correlation matrices[cite: 4728].

### [Problem Set 2: Optimization & Advanced Architectures](./Problem_Set_2)
Explores second-order optimization, model compression, and specialized neural architectures.
* [cite_start]**Advanced Optimizers:** * **Newton’s Method:** Convergence analysis and error rates[cite: 5788].
    * [cite_start]**Conjugate Gradient:** Minimizing quadratic functions[cite: 5791].
    * [cite_start]**Adadelta:** Implementation and visualization of trajectories on rotated surfaces[cite: 5808].
    * [cite_start]**Momentum:** Stability analysis of Steepest Descent with momentum[cite: 5816].
* [cite_start]**Model Compression:** Weight pruning strategies (Magnitude vs. Random) and "Super-weight" analysis[cite: 5797].
* [cite_start]**CNNs:** Convolution vs. Matrix Multiplication (Toeplitz matrices), pooling commutativity, and parameter counting[cite: 5842].
* [cite_start]**RBF Networks:** Designing Radial Basis Function networks for function approximation[cite: 5857].
* [cite_start]**Unsupervised Learning:** Self-Organizing Maps (SOM) and Learning Vector Quantization (LVQ)[cite: 5868].

### [Problem Set 3: RNNs & Fuzzy Logic](./Problem_Set_3)
Covers temporal sequence modeling and fuzzy set theory.
* [cite_start]**Recurrent Neural Networks:** * Modeling Auto-Regressive (AR) processes[cite: 5661].
    * [cite_start]Implementation of GRU/LSTM for time-series prediction[cite: 3940].
* [cite_start]**LSTM Mechanics:** Analysis of vanishing gradients, gates ($f_t, i_t, o_t$), and cell state propagation[cite: 5669].
* [cite_start]**Model Reduction (CNNs):** Kernel pruning based on $L_1$ norms and Cosine Similarity[cite: 5641].
* [cite_start]**Fuzzy Logic:** * Operations on Fuzzy Subsets (Intersection, Union, Complement, Algebraic Sum)[cite: 5693].
    * [cite_start]Fuzzy De Morgan Laws and Kleene-Dienes implication[cite: 5722].
    * [cite_start]Linguistic Hedges ("Very", "More or Less") and fuzzy subset theory[cite: 5753].

## 🛠️ Technologies
* **MATLAB**: Core implementation of optimization algorithms (Gradient Descent, Adadelta, LMS) and Neural Networks (MLP, RBF, ADALINE).
* **Python**: Used for specific RNN/LSTM tasks and data generation.
* **LaTeX**: Used for typesetting mathematical proofs and reports.

## 📊 Visualizations
*(You should upload the best images from your reports to a 'figures' folder and link them here. For example:)*

| Widrow-Hoff Learning Curve | Adadelta Trajectory |
|:--------------------------:|:-------------------:|
| <img src="./Problem_Set_1/figures/sse_plot.png" width="400"> | <img src="./Problem_Set_2/figures/adadelta_plot.png" width="400"> |

## 📜 License
This project is for educational purposes. Please attribute if using the code.
