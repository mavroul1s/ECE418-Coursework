% Contour plot of f(x,y) = x^4 + y^4 - 4*x.*y + 1
clear; close all; clc

x = linspace(-2,2,400);
y = linspace(-2,2,400);
[X,Y] = meshgrid(x,y);

F = X.^4 + Y.^4 - 4.*X.*Y + 1;

% contour levels (include level -1)
levels = linspace(-2,6,36);

figure('Units','normalized','Position',[0.1 0.1 0.6 0.6])
contourf(X,Y,F,levels,'LineColor','none')   % filled contours
hold on
%contour lines and label some
[CC,hc] = contour(X,Y,F,[-1 0 1 2 4],'LineColor','k','LineWidth',1);
clabel(CC,hc,'FontSize',10,'Color','k')

% highlight the global-min contour f = -1
contour(X,Y,F,[-1 -1],'LineColor','r','LineWidth',2)

crit = [0 0; 1 1; -1 -1];
plot(crit(:,1),crit(:,2),'ko','MarkerFaceColor','y','MarkerSize',8)
text(crit(1,1)+0.05,crit(1,2)+0.05,'(0,0)','FontSize',11)
text(crit(2,1)+0.05,crit(2,2)+0.05,'(1,1)','FontSize',11)
text(crit(3,1)+0.05,crit(3,2)+0.05,'(-1,-1)','FontSize',11)

axis equal
xlim([-2 2]); ylim([-2 2]);
xlabel('x'); ylabel('y')
title(' f(x,y)=x^4+y^4-4xy+1')
colorbar
hold off
