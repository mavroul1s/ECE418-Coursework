## SECTION 1: Model Reduction on CNNS

### Problem-01
You are given the following three kernels (filters) in a convolutional layer of a CNN.

**Filter 1:**
$$\begin{bmatrix} -0.3 & -0.2 & -0.4 \\ 0.7 & 0.5 & 0.6 \\ 0.5 & 0.1 & 0.7 \end{bmatrix}$$

**Filter 2:**
$$\begin{bmatrix} 0.5 & 0.7 & 0.6 \\ -0.1 & -0.3 & -0.3 \\ 0.1 & 0.4 & 0.4 \end{bmatrix}$$

**Filter 3:**
$$\begin{bmatrix} 0.1 & 0.4 & 0.3 \\ 0.2 & 0.8 & 0.4 \\ -0.5 & -0.4 & -0.8 \end{bmatrix}$$

Suppose you are asked to prune one of these filters in order to reduce the model size of the CNN, e.g., for purposes of running a TinyML application.
* A. Which one would you prune based on the $L_{t}$ norm?
* B. Which one would you prune based on some similarity measure among the three kernels? Define your similarity measure. For instance, you may get inspiration from matrix similarity concepts (each kernel is a matrix/tensor after all).

For both [A] and [B] show you analytic calculations.

---

## SECTION 2: Recurrent neural networks

### Problem-02
You are asked to generate an Auto Regressive (AR) model and then create an RNN (such as LSTM, or GRU) that predicts it. Generate samples of an Auto Regressive model of the form:

$X_{t}=a_{1}X_{t-1}+a_{2}X_{t-2}+a_{3}X_{t-3}+a_{4}X_{t-4}+U_{t}$

where $a_{1}=0.5$, $a_{2}=-0.25$, $a_{3}=0.1$, $a_{4}=-0.2$ and $U_{t}$ is independent-identically distributed Uniform in the interval (0, 0.05).
* A. Now design an RNN of your choice that predicts the sequence.
* B. Apply the training algorithm on new samples and calculate the averaged cost square error cost function. Investigate different number of training samples. (You are not allowed to use your model knowledge when you design the RNN.) Show the accuracy vs. training samples.

### Problem-03
Recall the elements of a module in an LSTM and the corresponding computations, where $\odot$ stands for point-wise multiplication:

$f_{t}=\sigma(W_{f}h_{t-1}+U_{f}x_{t})$
$i_{t}=\sigma(W_{i}h_{t-1}+U_{i}x_{t})$
$o_{t}=\sigma(W_{o}h_{t-1}+U_{o}x_{t})$
$\bar{C}_{t}=\tanh(W_{c}h_{t-1}+U_{c}x_{t})$
$C_{t}=f_{t}\odot C_{t-1}+i_{t}\odot\bar{C}_{t}$
$h_{t}=o_{t}\odot\tanh(C_{t})$

* A. What do the gates $f_{t}$, $i_{t}$ and $o_{t}$ do?
* B. Which of the quantities next to the figure are always positive?

Let's now try to understand how this architecture approaches the vanishing gradients problem. To calculate the gradient $\partial L/\partial\theta$, where $\theta$ stands for the parameters $(W_{f},W_{i},W_{o},W_{c})$, we now have to consider the cell state $C_{t}$ instead of $h_{t}$ (see course slides). Like $h_{t}$ in normal RNNs, $C_{t}$ will also depend on the previous cell states $C_{t-1},...,C_{0}$ so we get a formula of the form:

$$\frac{\partial L}{\partial W}=\sum_{t=0}^{T}\sum_{k=1}^{t}\frac{\partial L}{\partial C_{t}}\frac{\partial C_{t}}{\partial C_{k}}\frac{\partial C_{k}}{\partial W}$$

C. We know that $\frac{\partial C_{t}}{\partial C_{k}}=\prod_{i=k+1}^{t}\frac{\partial C_{i}}{\partial C_{i-1}}$. Let $f_{t}=1$ and $i_{t}=0$ such that $C_{t}=C_{t-1}$ for all t. What is the gradient $\partial C_{t}/\partial C_{k}$ in this case?

---

## SECTION 4: Fuzzy subsets theory

### Problem-04
Consider the following reference set: $\{A, B, C, D, E, F, G\}$, and the fuzzy subsets:
$\mathcal{A}=\{(A|0),(B|0.3),(C|0.7),(D|1),(E|0),(F|0.2),(G|0.6)\}$
$\mathcal{B}=\{(A|0.3),(B|1),(C|0.5),(D|0.8),(E|1),(F|0.5),(G|0.6)\}$
$\mathcal{C}=\{(A|1),(B|0.5),(C|0.5),(D|0.2),(E|0),(F|0.2),(G|0.9)\}$

Calculate the following:
* A. $A\cap B$
* B. $A\cup B$
* C. $A\cap B^{c}$
* D. $(A\cup B^{c})\cap C$
* E. $(A\cap B)^{c}\cup C^{c}$
* F. $(A\cap A)\cup A$

### Problem-05
Considering the three fuzzy subsets of Problem-04, calculate:
* A. $A+B+C$ (algebraic sum)
* B. $A.(B\hat{+}C)$

and prove:
* C. $A\subset A$ and $A\hat{+}A\supset A$
* D. $C\supset A.(B\hat{+}C)$
* E. $A.B \subset B\hat{+}A$

### Problem-06
Give the power set of fuzzy subsets for the following cases:
* A. $E=\{x_{1},x_{2}\}$, $M=\{0,1/3,2/3,1\}$
* B. $E=\{x_{1},x_{2},x_{3}\}$, $M=\{a,b,c\}$, $a<b<c$

### Problem-07
Prove the fuzzy DeMorgan laws:
* A. $X\cap Y=(X^{c}\cup Y^{c})^{c}$
* B. $X\cup Y=(X^{c}\cap Y^{c})^{c}$

### Problem-08
Dimitris and Fany go to park if it is a beautiful day and it is not too hot, or if it isn't raining. Assuming that:
* It is a beautiful day with 0.6 degree
* It is hot with 0.4 degree
* It is raining with 0.8 degree

With which degree Dimitris and Fany will go to park?

### Problem-09
Let $P(x)$ and $Q(x)$ be fuzzy truth functions, each of which can only give truth values of 0, 0.5 and 1. That is, for all x, $P(x)$ is in the set $\{0, 0.5, 1\}$ and $Q(x)$ is in the set $\{0, 0.5, 1\}$. Recall the Kleene-Dienes definition of implication: "$a\rightarrow b$" is equivalent to "(not a) or b".

Compute the truth table for the fuzzy statement "$(P(x) \text{ and } (P(x)\rightarrow Q(x)))\rightarrow Q(x)$". How does this compare for the same truth table in crisp logic (where $P(x)$ and $Q(x)$ can only be "true" or "false")?

### Problem-10
Assume that the truth function of $A(x)$ is the following:
* $A(x)=1$ for $x \le 2$
* $A(x)=1-(x-2)/3$ for $2 < x < 5$
* $A(x)=0$ for $x \ge 5$

and $B(x)$ has the following:
* $B(x)=0$ for $x \le 3$
* $B(x)=(x-3)/4$ for $3 < x < 7$
* $B(x)=1$ for $x \ge 7$

Find for which values of x, the following statement has the maximum truth value: "not($A(x)$ OR $B(x)$)"

### Problem-11
"Very" is used as an adjective to reduce vagueness on fuzzy set membership. The interpretation is that if the statement "A is true" has truth value equal to x, then the statement "A is very true" has truth value $x^{2}$, because the "very true" is more demanding.
* A. True or false (explain your answer): Let "S" be a fuzzy set. Then "Very S" is a fuzzy subset of "S".

"More or less" is used as an adjective to increase vagueness - the interpretation is that if "A is true" has truth value x, then "A is more or less true" has truth value $\sqrt{x}$.
* B. True or false (explain your answer): Let "S" be a fuzzy set. Then "S" is a fuzzy subset of "more or less S".
* C. Using the definitions just given, is it true that "not very S" is a subset of "more or less S", or vice versa, or is it impossible to say?
* D. Is "not more or less S" a subset of "very S", or vice versa, or is it impossible to say?
