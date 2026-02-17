A = [2 0 0; 0 10 0; 0 0 6]; 
b = [0; 0; 0];
x = [2; 2; 2];  

r = b - A*x;    % Initial residual
d = r;          % Initial direction
points = x;     % Store points for plotting


for i = 1:3 
    alpha = (r' * r) / (d' * A * d);   % Step size
    x = x + alpha * d;                 % Update position
    points = [points, x];              % Save for plot
    
    r_new = r - alpha * A * d;         % Update residual
    
    if norm(r_new) < 1e-10
        break; 
    end
    
    beta = (r_new' * r_new) / (r' * r); % Update direction scalar
    d = r_new + beta * d;              % New conjugate direction
    r = r_new;                        
end

disp('Minimum found at x:');
disp(x);

figure;
plot3(points(1,:), points(2,:), points(3,:), '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on; box on;
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Path of Conjugate Gradient Method to Minimum');
view(45, 30); % Adjust camera angle for better view