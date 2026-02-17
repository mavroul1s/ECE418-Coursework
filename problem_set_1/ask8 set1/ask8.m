clear; close all; clc;

% Data
p1 = [1;2]; t1 = -1;
p2 = [-2;1]; t2 = 1;

R = 0.5*(p1*p1' + p2*p2');    % = 2.5*I
r_pt = 0.5*(p1*t1 + p2*t2);   % = [-1.5; -0.5]
w_star = R \ r_pt;            % = [-0.6; -0.2]

%% Contour plot of MSE
w1 = linspace(-1.5,0.5,300);
w2 = linspace(-1.2,1.2,300);
[W1,W2] = meshgrid(w1,w2);
J = zeros(size(W1));
for i = 1:numel(W1)
    w = [W1(i); W2(i)];
    e1 = t1 - w' * p1;
    e2 = t2 - w' * p2;
    J(i) = 0.5*(e1^2 + e2^2);
end

figure(1);
contour(W1,W2,J,30); axis equal; hold on
plot(w_star(1), w_star(2), 'kp','MarkerFaceColor','k','MarkerSize',10)
text(w_star(1), w_star(2),'  w^*','FontWeight','bold')
xlabel('w_1'); ylabel('w_2');
title('A) MSE contours (weight-space)')

%%LMS trajectory 
w = [0;1];                 % initial weight W(0)
mu = 0.02;                 % very small learning rate (adjustable)
steps = 100;               % number of pattern updates
trajectory = zeros(2,steps+1);
trajectory(:,1) = w;
for k = 1:steps
    idx = mod(k-1,2)+1;   
    if idx==1, p=p1; t=t1; else p=p2; t=t2; end
    e = t - w'*p;
    w = w + mu * p * e;
    trajectory(:,k+1) = w;
end
plot(trajectory(1,:), trajectory(2,:), '-o', 'MarkerSize',3, 'LineWidth',1);
plot(trajectory(1,1), trajectory(2,1), 'go','MarkerFaceColor','g','MarkerSize',8)
text(trajectory(1,1), trajectory(2,1),'  w(0)')
legend('MSE contours','w^*','LMS path','Location','northeastoutside')
hold off

%%patterns and decision boundaries
figure(2); hold on; grid on; axis equal
plot(p1(1),p1(2),'ro','MarkerFaceColor','r','MarkerSize',10); text(p1(1),p1(2),'  p_1 (t=-1)')
plot(p2(1),p2(2),'bo','MarkerFaceColor','b','MarkerSize',10); text(p2(1),p2(2),'  p_2 (t=+1)')

x = linspace(-3,3,200);
% boundary for w* : y = -3x
y_star = -3*x;
plot(x,y_star,'b-','LineWidth',1.5)
text(1, -3*1, '  decision boundary (w^*)')

% initial boundary w(0) = [0;1] -> y = 0
plot(x, zeros(size(x)), '--','LineWidth',1)
text(0.6, 0, '  decision boundary (w(0))')

xlabel('p_1 (x)'); ylabel('p_2 (y)');
title(' patterns and decision boundaries (no bias)')
xlim([-3 3]); ylim([-3 3]); hold off

%% Optional: plot J along iterations
J_traj = zeros(1,size(trajectory,2));
for i=1:size(trajectory,2)
    wtmp = trajectory(:,i);
    e1 = t1 - wtmp' * p1;
    e2 = t2 - wtmp' * p2;
    J_traj(i) = 0.5*(e1^2 + e2^2);
end
figure(3);
plot(0:steps, J_traj, '-o'); xlabel('Iteration (pattern updates)'); ylabel('MSE J(w)');
title('MSE along LMS trajectory');
grid on;
