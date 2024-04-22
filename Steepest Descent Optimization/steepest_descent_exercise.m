% This is an exercise to implement the Steepest Descent algorithm. 
% Jacob Bado, 3-16-24

% Define initial point
x0 = [0; 0; 0];

% Define analytically determined minimizer
x_min = [5; 3; -7];

% Define error threshold
err_tol = 1e-6;

% Implement steepest descent algorithm
[x, l2x_x1, fx, l2dfx, l2x_xmin] = steepest_descent(x0, x_min, err_tol);

%% Plot results
fig = figure;
fig.Color = [1,1,1];
sgtitle('Results of Steepest Descent Optimization', 'FontWeight', 'bold')

subplot(2,2,1)
plot(fx, 'k.', 'MarkerSize', 5)
ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
ax.XGrid = 'on';
xlim('tight')
ylim('padded')
xlabel('Iteration (k)')
ylabel('$\mathbf{{f}(\vec{x}_k)}$', ...
    'Interpreter', 'latex');
title('Objective Function Values')

subplot(2,2,2)
plot(l2x_x1, 'k.', 'MarkerSize', 5)
ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
ax.XGrid = 'on';
xlim('tight')
ylim('padded')
xlabel('Iteration (k to k+1)')
ylabel('$\| \mathbf{\vec{x}_{k+1}} - \mathbf{\vec{x}_k} \|_2$', ...
    'Interpreter', 'latex')
title('Distance Between Consecutive Points')

subplot(2,2,3)
plot(l2dfx, 'k.', 'MarkerSize', 5)
ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
ax.XGrid = 'on';
xlim('tight')
ylim('padded')
xlabel('Iteration (k)')
ylabel('$\| \nabla \mathbf{f}(\mathbf{\vec{x}}_k) \|_2$', ...
    'Interpreter', 'latex')
title('l2 Norm of the Gradient Vector')

subplot(2,2,4)
plot(l2x_xmin, 'k.', 'MarkerSize', 5)
ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
ax.XGrid = 'on';
ax.YTick = [0, 2, 4, 6, 8, 10];
xlim('tight')
ylim('padded')
xlabel('Iteration (k)')
ylabel('$\| \mathbf{\vec{x}}_k - \mathbf{\vec{x}}^* \|_2$', ...
    'Interpreter', 'latex')
title('Error Between Each Point and the Minimizer')

print(gcf, 'Q3cFig.png', '-dpng', '-r500');

%% Show linear convergence

fig = figure;
fig.Color = [1,1,1];
plot(log(l2dfx), 'k.', 'MarkerSize', 5)
ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
ax.XGrid = 'on';
xlim('tight')
ylim('padded')
xlabel('Iteration (k)')
ylabel('$log(\| \nabla \mathbf{f}(\mathbf{\vec{x}}_k) \|_2)$', ...
    'Interpreter', 'latex')
title('Log of l2 Norm of the Gradient Vector')

print(gcf, 'Q3cLogFig.png', '-dpng', '-r500');