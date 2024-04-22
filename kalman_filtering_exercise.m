% Implementation of the Kalman Filter for Discrete Dynamical Systems, 
% Jacob Bado, 3-5-24

% (a) Generate observations
% Define time vector
t = 1:1000;

% Generate initial conditions
m0 = [1; 1];
C0 = [0.3, 0; 0, 0.6];
v0 = mvnrnd(m0, C0)';

% Define state transition matrix
M = [-0.01, -0.02; 0, -0.03];

% Define observation matrix
H = [1, 0];

% Define process noise covariance
Q = [0.05, 0; 0, 0.025];

% Define measurement noise variance
R = 0.025;

% Preallocate memory
v = zeros(2, length(t));
y = zeros(1, length(t));
xi = zeros(2, length(t));
eta = zeros(1, length(t));

% Compute results for j = 1
v(:, 1) = expm(M)*v0;
y(1) = H*v(:, 1);
xi(:, 1) = mvnrnd([0; 0], Q); 
eta(1) = mvnrnd(0, 0.025);

% Compute results for j = 2 to 1000
for j = 1:max(t)-1
    v(:, j+1) = expm(M*(t(j+1)-t(j)))*v(:, j);
    y(j+1) = H*v(:, j+1);
    xi(:, j+1) = mvnrnd([0; 0], Q); 
    eta(j+1) = mvnrnd(0, R);
end

% Add noise to system trajectory and observations
v = v + xi;
y = y + eta;

%% (b) Implement Kalman Filter

% Preallocate memory
m_hat = cell(1, length(t));
C_hat = cell(1, length(t));
d = cell(1, length(t));
S = cell(1, length(t));
K = cell(1, length(t));
m = cell(1, length(t));
C = cell(1, length(t));

% Define identity matrix
I = diag([ones(1, height(v))]);

% Compute prediction for j = 1
m_hat{1} = M*m0;
C_hat{1} = M*C0*M' + Q;

% Implement Kalman filter
for j = 0:max(t)-1
    if j > 0
        % Prediction
        m_hat{j+1} = M*m{j};
        C_hat{j+1} = M*C{j}*M' + Q;
    end
    % Innovation
    d{j+1} = y(:, j+1) - H*m_hat{j+1};
    
    % Covariance of innovation
    S{j+1} = H*C_hat{j+1}*H' + R;
    
    % Kalman gain
    K{j+1} = C_hat{j+1}*H'*S{j+1}^-1;
    
    % Analysis
    m{j+1} = m_hat{j+1} + K{j+1}*d{j+1};
    C{j+1} = (I - K{j+1}*H)*C_hat{j+1};
end

%% (c) Plot resulting predicted and updated values

% Preallocate memory
pred_var1 = zeros(1, length(t));
pred_var2 = zeros(1, length(t));
upd_var1 = zeros(1, length(t));
upd_var2 = zeros(1, length(t));
pred_var1_std = zeros(1, length(t));
pred_var2_std = zeros(1, length(t));

% Pull out variables and stds and store into vectors
for i = 1:length(t)
    pred_var1(i) = m_hat{i}(1);
    pred_var2(i) = m_hat{i}(2);
    upd_var1(i) = m{i}(1);
    upd_var2(i) = m{i}(1);
    pred_var1_std(i) = sqrt(C_hat{i}(1,1));
    pred_var2_std(i) = sqrt(C_hat{i}(2,2));
end

% Compute 2*std band
std_band1 = [pred_var1 + 2*pred_var1_std; pred_var1 - 2*pred_var1_std];
std_band2 = [pred_var2 + 2*pred_var2_std; pred_var2 - 2*pred_var2_std];


fig = figure;
fig.Color = [1,1,1];
sgtitle('Results of Kalman Filtering')
% Plot predicted
subplot(2,1,1)
plot(upd_var1, 'Color', [0,0.5,0.6]); hold on
plot(pred_var1, 'LineWidth', 2, 'Color', [0.5,0,0])
plot(y, 'k.', 'MarkerSize', 4)
plot(std_band1(1,:), '--', 'LineWidth', 2, 'Color', [0.5,0.5,0.5])
plot(std_band1(2,:), '--', 'LineWidth', 2, 'Color', [0.5,0.5,0.5])

ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('j')
ylabel('State 1')
title('State Variable 1')
legend('Updated Values', 'Predicted Values', 'Observations','2\sigma Band')

% Plot updated
subplot(2,1,2)
plot(upd_var2, 'Color', [0,0.5,0.6]); hold on
plot(pred_var2, 'LineWidth', 2, 'Color', [0.5,0,0])
plot(std_band2(1,:), '--', 'LineWidth', 2, 'Color', [0.5,0.5,0.5])
plot(std_band2(2,:), '--', 'LineWidth', 2, 'Color', [0.5,0.5,0.5])

ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('j')
ylabel('State 2')
title('State Variable 2')
legend('Updated Values', 'Predicted Values', '2\sigma Band')

%% (d) Evaluate accuracy

mse = mean((y - pred_var1).^2);
cor = corr(y', pred_var1');