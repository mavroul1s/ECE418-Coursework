clear; clc; close all;

P = [ 0   0;
      0   1;
      1   0;
      2   0;
     -1  -1;
      0  -2.5;
      1.5 -1.5 ];

N = size(P,1);

T1 = [-1; -1; -1; -1; -1; -1; -1];
T1(1:2) = 1;   

T2 = -1 * ones(N,1);
T2(5:7) = 1;  

w1 = (P' * P) \ (P' * T1);
w2 = (P' * P) \ (P' * T2);

disp('Weights for ADALINE 1 (Class I vs Rest):');
disp(w1);

disp('Weights for ADALINE 2 (Class III vs Rest):');
disp(w2);

figure; hold on;

% Plot patterns
plot(P(1:2,1), P(1:2,2), 'ro', 'MarkerSize', 8);  % Class I
plot(P(3:4,1), P(3:4,2), 'go', 'MarkerSize', 8);  % Class II
plot(P(5:7,1), P(5:7,2), 'bo', 'MarkerSize', 8);  % Class III

% (ADALINE 1)
x = linspace(-4,4,200);
x_line1 = -(w1(1)/w1(2)) * x;
plot(x, x_line1, 'r', 'LineWidth', 2);

% (ADALINE 2)
x_line2 = -(w2(1)/w2(2)) * x;
plot(x, x_line2, 'b', 'LineWidth', 2);

title('Two Decision Boundaries for 3-Class ADALINE');
xlabel('x_1'); ylabel('x_2');
axis equal; grid on;
legend('Class I','Class II','Class III','Boundary 1','Boundary 2');
hold off;

w1_range = linspace(w1(1)-5, w1(1)+5, 200);
w2_range = linspace(w1(2)-5, w1(2)+5, 200);
[W1_mesh, W2_mesh] = meshgrid(w1_range, w2_range);

MSE1 = zeros(size(W1_mesh));

for i = 1:numel(W1_mesh)
    w = [W1_mesh(i); W2_mesh(i)];
    y = P * w;
    err = T1 - y;
    MSE1(i) = mean(err.^2);
end

figure;
contour(W1_mesh, W2_mesh, MSE1, 30);
hold on;
plot(w1(1), w1(2), 'rx','MarkerSize',12,'LineWidth',2);
title('MSE Contour Plot — ADALINE 1');
xlabel('w_1'); ylabel('w_2');
grid on;

