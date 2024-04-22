% Compute the linear quadratic regulator to modulate a dummy control system. 
% Jacob Bado, 4-1-24

% (a) 

% Define state matrix
A = [6, -1, 0; 0, 1, 3; 5, 0, -1];

% Define input vector
B = [1; 2; 0];

% Define output vector
C = [1, 0, 0];

% Define system
dx = @(x, u) A*x + B*u;

% Define output equation
y = @(x) C*x;

% Compute controllability matrix
Wc = [B, A*B, A^2*B];

% Compute observability matrix
Wo = [C; C*A; C*A^2];

% Assess if Wc and Wo are full rank to determine system controllability and
% observability
det_Wc = det(Wc);
det_Wo = det(Wo);

%% (b)

% Define Q and R
Q = diag([1, 1, 1]);
R = 1;

% Define the algebraic riccati equation
riccati = @(P) P*A + A'*P - P*B*R^-1*B'*P + Q;

% Define initial conditions for optimization
P0 = ones(3, 3);

% Solve the riccati equation
P = fsolve(riccati, P0);

% Compute K
K = R\(B'*P);

%% (c)
 
% Define time span
dt = 0.2;
t = dt:dt:5;

% Define initial conditions
x0 = [1, 1, 1]';

% Preallocate memory
x = zeros(length(x0), length(t));

% Compute analytic solution for the system over the time span
for i = 1:length(t)
    x(:, i) = exp(A*(t(i) - 0))*x0;
end

% Plot results
fig = figure;
fig.Color = [1, 1 ,1];
sgtitle('Solutions of Control System', 'FontWeight', 'bold')
for i = 1:3
    subplot(3, 1, i)
    plot(t, x(i, :), 'k-', 'LineWidth', 1.5)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlabel('Time')
    ylabel('Value')
    title(['State: ', num2str(i)])
    xlim('tight')
    ylim('tight')
end

%% (d)

% Compute analytic solution for the system over the time span
for i = 1:length(t)
    x(:, i) = exp((A - B*K)*(t(i) - 0))*x0;
end

% Plot results
fig = figure;
fig.Color = [1, 1 ,1];
sgtitle('Solutions of Control System', 'FontWeight', 'bold')
for i = 1:3
    subplot(3, 1, i)
    plot(t, x(i, :), 'k-', 'LineWidth', 1.5)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlabel('Time')
    ylabel('Value')
    title(['State: ', num2str(i)])
    xlim('tight')
    ylim('tight')
end