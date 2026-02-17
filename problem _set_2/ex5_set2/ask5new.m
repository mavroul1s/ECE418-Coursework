clc; clear; close all;

problems = [0.4, 1;   3.0, 1;   3.0, 2]; 
w0 = [-8; 6]; rho = 0.9; eps = 1e-6; N = 1000;

for p = 1:3
    alpha = problems(p, 1); mode = problems(p, 2);
    
    if mode == 1
        F = @(w) 0.1*w(1)^2 + 2*w(2)^2;
        G = @(w) [0.2*w(1); 4*w(2)];
        title_str = sprintf('Prob %d: Alpha=%.1f (Normal)', p, alpha);
    else
        F = @(w) 0.1*(w(1)+w(2))^2 + 2*(w(1)-w(2))^2;
        G = @(w) [4.2*w(1)-3.8*w(2); -3.8*w(1)+4.2*w(2)];
        title_str = sprintf('Prob %d: Rotated', p);
    end

    w = w0; Eg2 = 0; Edx2 = 0; 
    path = w; path_z = F(w); % Store history
    
    for i = 1:N
        g = G(w);
        Eg2 = rho * Eg2 + (1 - rho) * g.^2;             % Accumulate Gradient
        dx  = -(sqrt(Edx2 + eps) ./ sqrt(Eg2 + eps)) .* g; % Compute Step
        w   = w + alpha * dx;                           % Update Weights
        Edx2 = rho * Edx2 + (1 - rho) * dx.^2;  
        path = [path, w]; path_z = [path_z, F(w)];      % Save path
    end

    figure('Name', title_str); set(gcf, 'Position', [100, 100, 900, 400]);
    [X, Y] = meshgrid(linspace(-12, 12, 50));
    Z = arrayfun(@(x,y) F([x;y]), X, Y); % Calculate surface height
    
    subplot(1,2,1); contour(X,Y,Z,20); hold on; grid on;
    plot(path(1,:), path(2,:), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('2D Trajectory'); xlabel('w_1'); ylabel('w_2');
    
    subplot(1,2,2); mesh(X,Y,Z,'FaceAlpha',0.3); hold on; grid on;
    plot3(path(1,:), path(2,:), path_z, 'r-o', 'LineWidth', 2, 'MarkerFaceColor','r');
    title('3D Projection'); view(45, 30);
end