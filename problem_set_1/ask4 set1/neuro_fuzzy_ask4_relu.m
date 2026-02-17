p = linspace(-2,2,100);
w1 = -2;
w2 = -1;
b1 = -2;
b2 = -0.2;
w22 = 2.5;
w11 = 1.5;
b3 = 1;

% Hidden layer
n1 = w1*p + b1;
n2 = w2*p + b2;

% ReLU activation
a1 = max(0, n1);
a2 = max(0, n2);

% Output layer
n3 = a1*w11 + a2*w22 + b3;

% Pure linear activation
a3 = n3;

% Plotting
figure;

subplot(3,1,1);
plot(p, a1, 'b', 'LineWidth', 2);
title('ReLU a1');
xlabel('p');
ylabel('a1');
grid on;

subplot(3,1,2);
plot(p, a2, 'r', 'LineWidth', 2);
title('ReLU a2');
xlabel('p');
ylabel('a2');
grid on;

subplot(3,1,3);
plot(p, a3, 'k', 'LineWidth', 2);
title('Purelin a3');
xlabel('p');
ylabel('a3');
grid on;
