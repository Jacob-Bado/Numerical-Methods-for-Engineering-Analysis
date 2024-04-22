% This is an exercise to model a linear dynamical system representing a
% 6 compartment pharmacokinetic model and implements optimization to 
% control the system.
% Jacob Bado, 4-20-24

% x(1) = brain, x(2) = stomach, x(3) = intestines, 
% x(4) = heart, x(5) = lungs,   x(6) = other

% (a) Define matrix of rate constants
M = [ 0.6,  0.1,    0,  0.1,    0,    0; ...
        0,  0.5,    0,  0.1,    0,    0; ...
        0,  0.2,  0.5,    0,    0,    0; ...
      0.4,  0.2,    0,  0.5,  0.6,  0.3; ...
        0,    0,    0,  0.2,  0.4,    0; ...
        0,    0,    0,  0.1,    0,  0.7];

% Define IV drip initial condition
x_drip = 0.1;

% Define target concentrations
x_target = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1];

% Define number of time steps (1 hour)
t_steps = 12;

% Preallocate memory, x0 is defined as the first column
x_drips = zeros(t_steps, 1);
x = zeros(height(M), t_steps + 1);

% Define optimizer options
opts = optimoptions(@fmincon, 'MaxIterations',10000, ...
    'OptimalityTolerance', 1e-10);

for i = 1:t_steps
% Define objective function
    obj_fxn = @(x_drip) ...
        norm(((M*x(:, i) + [0; 0; 0; x_drip; 0; 0]) - x_target)).^2;

    % Determine optimal IV drip, constrain such that IV drip is nonnegative
    x_drip = fmincon(obj_fxn, x_drip, [], [], [], [], 0, Inf, [], opts);

    % Track IV drip
    x_drips(i) = x_drip;

    % Move the dynamical system forward in time
    x(:, i+1) = M*x(:, i) + [0; 0; 0; x_drip; 0; 0];
end

% Plot results
fig = figure; hold on
fig.Color = [1,1,1];
for i = 1:height(x)
    plot(0:5:t_steps*5, x(i, :), 'LineWidth', 2)
end
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
box on
title('Time Series of Compartmental Model')
xlabel('Time (min)')
ylabel('Drug (mg)')
legend('Brain', 'Stomach', 'Intestines', 'Heart', 'Lungs', 'Other', ...
    'Location', 'east')

%% (b) Implement Tikhonov regularization

% Initialize previous IV drip value for the regularization term
x_drip_prev = 0;

% Define the regularization parameter
lambda = 1;

for i = 1:t_steps
    % Define the objective function with Tikhonov regularization
    obj_fxn = @(x_drip) ...
        norm(((M*x(:, i) + [0; 0; 0; x_drip; 0; 0]) - x_target)).^2 + ...
        lambda*(x_drip - x_drip_prev).^2;

    % Determine optimal IV drip, constrain such that IV drip is nonnegative
    x_drip = fmincon(obj_fxn, x_drip, [], [], [], [], 0, Inf, [], opts);

    % Update the dynamical system
    x(:, i+1) = M*x(:, i) + [0; 0; 0; x_drip; 0; 0];

    % Update previous IV drip value for the next iteration
    x_drip_prev = x_drip;
end

% Plot results
fig = figure; hold on
fig.Color = [1,1,1];
for i = 1:height(x)
    plot(0:5:t_steps*5, x(i, :), 'LineWidth', 2)
end
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
box on
title(sprintf('Time Series of Compartmental Model\nWith Tikhonov Regularization'))
xlabel('Time (min)')
ylabel('Drug (mg)')
legend('Brain', 'Stomach', 'Intestines', 'Heart', 'Lungs', 'Other', ...
    'Location', 'east')