clc; clear; close all;

% -------- Given Parameters --------
w11_1 = -0.27;
w21_1 = -0.41;
b1_1  = -0.48;
b2_1  = -0.13;

w11_2 = 0.09;
w12_2 = -0.17;
b2    = 0.48;

alpha = 0.10;

% -------- Input Range --------
p = linspace(-2, 2, 400);

% -------- Layer 1 Net Inputs --------
n1_1 = w11_1 .* p + b1_1;
n2_1 = w21_1 .* p + b2_1;

% -------- Swish Activation --------
sigmoid = @(x) 1 ./ (1 + exp(-x));
a1_1 = n1_1 .* sigmoid(n1_1);
a2_1 = n2_1 .* sigmoid(n2_1);

% -------- Layer 2 Net Input --------
n2 = w11_2 .* a1_1 + w12_2 .* a2_1 + b2;

% -------- Leaky ReLU Activation --------
a2 = max(0, n2) + alpha .* min(0, n2);

% -------- Plotting --------
figure;

subplot(3,2,1);
plot(p, n1_1, 'LineWidth', 1.5);
title('n_1^1 vs p'); grid on;

subplot(3,2,2);
plot(p, a1_1, 'LineWidth', 1.5);
title('a_1^1 vs p'); grid on;

subplot(3,2,3);
plot(p, n2_1, 'LineWidth', 1.5);
title('n_2^1 vs p'); grid on;

subplot(3,2,4);
plot(p, a2_1, 'LineWidth', 1.5);
title('a_2^1 vs p'); grid on;

subplot(3,2,5);
plot(p, n2, 'LineWidth', 1.5);
title('n^2 vs p'); grid on;

subplot(3,2,6);
plot(p, a2, 'LineWidth', 1.5);
title('a^2 vs p'); grid on;

sgtitle('Neural Network Responses for -2 < p < 2');
