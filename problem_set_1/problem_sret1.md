## [cite_start]SECTION 0: Warming-up with linear algebra, calculus, and automatic control [cite: 8]

### [cite_start]Problem-01 [cite: 9]
[cite_start]Plot the contour lines of the following function: $f(x,y)=x^{4}+y^{4}-4xy+1$[cite: 10]. [cite_start]Then, find and characterize the real (local) minima/maxima of this function (Show your analytic calculations)[cite: 10].

### [cite_start]Problem-02 [cite: 11]
[cite_start]Execute two iterations of the Gradient Descent to the function $f(x,y)=6x^{2}-4xy+4y^{2}$ with initial point $x_{0}=(3,2)$[cite: 12]. [cite_start]Show your analytic calculations[cite: 12].

---

## [cite_start]SECTION 1: Introduction to neural networks [cite: 13]

### [cite_start]Problem-03 [cite: 14]
[cite_start]Express the derivative $dS/dx$, denoted as $S^{\prime}$, of the following activation functions $S$ in terms of the original function $S$, i.e., determine such that $S^{\prime}=\varphi(S,x)$[cite: 15]. [cite_start][The first three functions comprise established activation functions, with $c=1$ known as logsig, tansig, and Google's Swish, respectively.] [cite: 15]
* [cite_start]$S=\frac{1}{1+e^{-x}}$ [cite: 17]
* [cite_start]$S=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ [cite: 17]
* [cite_start]$S=\frac{x}{1+e^{-x}}$ [cite: 18]
* [cite_start]$S=x\times\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ [cite: 18]

### [cite_start]Problem-04 [cite: 20]
[cite_start]Consider the following neural network with ${w^{1}}_{1,1}=-2$, ${w^{1}}_{2,1}=-1$, $b_{1}^{1}=-2$, $b_{2}^{1}=0.2$, ${w^{2}}_{1,1}=1.5$, ${w^{2}}_{1,2}=2.5$, $b^{2}=1$[cite: 21, 41]. [cite_start]Sketch the following responses (plot the indicated variable versus $p$ for $-2<p<2$)[cite: 42]:
* A. $x^{1}$ [cite: 43]
* B. ${a^{1}}_{2}$ [cite: 44]
* C. $a^{2^{7}}$ [cite: 45]

Then, change the logsig activation function with the ReLU activation function and sketch again the aforementioned responses[cite: 46].

### Problem-05 [cite: 47]
Consider the following neural network with the following initialization: $w_{1,1}^{1}=-0.27$, $w_{2,1}^{1}=-0.41$, $b_{1}^{1}=-0.48$, $b_{2}^{1}=-0.13$, $w_{1,1}^{2}=0.09$, $w_{1,2}^{2}=-0.17$, $b^{2}=0.48$[cite: 49, 76, 77]. Sketch the following responses (plot the indicated variable versus $p$ for $-2<p<2$)[cite: 68]:
* i. $n_{1}^{1}$ [cite: 69]
* ii. $a_{1}^{1}$ [cite: 70, 71]
* iii. $n_{2}^{1}$ [cite: 72]
* iv. $a_{2}^{1}$ [cite: 73]
* v. $n^{2}$ [cite: 74]
* vi. $a^{2}$ [cite: 75]

---

## SECTION 2: Working with ADALINE neural networks [cite: 79]

### Problem-06 [cite: 80]
Suppose that you are given the following seven reference patterns and their categories: Class I consists of: $p_{1}=\{0,0\}$, $p_{2}=\{0,1\}$, Class II consists of: $p_{3}=\{1,0\}$, $p_{4}=\{2,0\}$, and Class III consists of: $p_{5}=\{-1,-1\}$, $p_{6}=\{0,-2.5\}$, $p_{7}=\{1.5,-1.5\}$[cite: 81]. The probability of each vector $p_{1}$, $p_{2}$ is 0.25, and the probability of each vector $p_{3}$, $p_{4}$, $p_{5}$, $p_{6}$, and $p_{7}$ is 0.1[cite: 82]. 
* A. Select appropriate target (category) values[cite: 83].
* B. Draw the network diagram for an ADALINE network with no bias that could be trained on these patterns[cite: 84].
* C. Sketch the contour plot of the mean square error performance index[cite: 85].
* D. Show the optimal decision boundary (for the weights that minimize mean square error), and verify that it separates the patterns into the appropriate categories[cite: 86].
* E. Find the maximum stable learning rate for the LMS algorithm[cite: 87]. Change the target values to opposite values, and see how this change affected the maximum stable learning rate? [cite: 88]

### Problem-07 [cite: 89]
Repeat the work of Widrow and Hoff on a pattern recognition problem from their classic 1960 paper (link-4 on the course's Webpage)[cite: 90]. They wanted to design a recognition system that would classify the six patterns shown below: Patterns T, G, F[cite: 91, 92, 93, 94, 95]. Targets 60, 0, -60[cite: 96]. These patterns represent the letters T, G and F, in an original form on the top and in a shifted form on the bottom[cite: 97]. The targets for these letters (in their original and shifted forms) are +60, 0 and -60, respectively[cite: 98]. The objective is to train a network so that it will classify the six patterns into the appropriate T, G or F groups[cite: 99]. The blue squares in the letters will be assigned the value +1, and the white squares will be assigned the value -1[cite: 100]. 

First we convert each of the letters into a single 16-element vector[cite: 101]. We choose to do this by starting at the upper left corner, going down the left column, then going down the second column, etc.[cite: 102]. Learning rate $\alpha=0.03$[cite: 103]. Present the training patterns in a random sequence[cite: 103]. You should use an ADALINE of the following topology: $a = purelin(Wp+b)$[cite: 103, 113]. You are required to draw a plot of the sum square error versus training steps[cite: 114]. Each step is defined as the presentation of one input pattern to the neural network[cite: 115]. [Your plot should look similar to that in Fig.5 of Widrow-Hoff original paper.] [cite: 116]

### Problem-08 [cite: 117]
Suppose that we have the following two reference patterns and their targets: P2 =[cite: 119, 121]. The vectors are equiprobable[cite: 122]. We want to train an ADALINE network without a bias on this data set[cite: 122].
* A. Sketch the contour plot of the mean square error performance index[cite: 123].
* B. Sketch the optimal decision boundary[cite: 124].
* C. Sketch the trajectory of the LMS algorithm on your contour plot[cite: 125]. Assume a very small learning rate, and start with initial weights $W(0)=[\begin{matrix}0&1\end{matrix}]$[cite: 126].

---

## SECTION 3: Basics of MLPs [cite: 127]

### Problem-09 [cite: 128]
Write code to implement the backpropagation algorithm for a $1-S^{1}-1$ MLP network (logsigmoid-linear)[cite: 129]. Write the program using matrix operations, as we did in the class lecture[cite: 130]. Choose the initial weights and biases to be random numbers uniformly distributed between -0.5 and 0.5, and train the network to approximate the function $g(p)=1+sin[p(\pi/3)]$ for $-2\le p\le2$[cite: 131, 132]. Use $S^{1}=2$, $S^{1}=6$, $S^{1}=10$ and $S^{1}=20$[cite: 133]. Experiment with several different values for the learning rate $\alpha$, and use several different initial conditions[cite: 133]. Discuss the convergence properties of the algorithm as the learning rate changes[cite: 134].

### Problem-10 [cite: 135]
The standard steepest descent backpropagation algorithm, which is summarized in the slide entitled "Summary of backpropagation algorithm" in Lecture-06, was designed to minimize the performance function that was the sum of squares of the network errors, as given in the last equation of slide 17 of Lecture-06[cite: 136]. Suppose that we want to change the performance function to the sum of the fourth powers of the errors $(e^{4})$ plus the sum of the squares of the weights and biases in the network[cite: 137]. Show how the equations in the slide entitled "Summary of backpropagation algorithm" will change for this new performance function[cite: 138]. (You don't need to rederive any steps which are already derived in our lectures and do not change.) [cite: 139]
