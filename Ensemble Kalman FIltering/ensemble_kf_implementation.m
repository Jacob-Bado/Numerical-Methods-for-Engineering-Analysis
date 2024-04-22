% This is an exercise to implement the Ensemble Kalman Filter for
% observations of a predator-prey model
% Jacob Bado, 3-17-24

% Load observations
load('data_table');
obs_t = data_table.measurement_time;
obs_x = data_table.prey_population_count;
obs_y = data_table.predator_population_count;

% Define ensemble size
N = 500;

% Define initial condition parameters
m0 = [100; 50];
C0 = [20, 0; 0, 10];

% Generate ensemble of initial conditions
z0_ens = mvnrnd(m0, C0, N)';

% Define number of ensembles
num_ensembles = length(obs_t)-1;

% Define s parameter
s = 1;

%  Define process noise covariance
Q = [10, 0; 0, 5];

% Define measurement noise variance
R = 10;

% Define observation operator
H = [1, 0];

% Define dynamical system model parameters
par = [0.1, 0.002; 0.0025, 0.2];

% Implement ensemble Kalman filter function
[m_hat, C_hat, upd_z_ens] = ensemble_kf_preypred...
    (z0_ens, num_ensembles, N, s, Q, R, H, obs_t, obs_x, par);

% MSE and correlation between predicted means and measurements
mse_pred_x = mean((m_hat(:, 1) - obs_x(2:end)).^2);
corr_pred_x = corr(m_hat(:, 1), obs_x(2:end));

mse_pred_y = mean((m_hat(:, 2) - obs_y(2:end)).^2);
corr_pred_y = corr(m_hat(:, 2), obs_y(2:end));

% MSE and correlation between updated ensemble means and measurements
mean_upd_x = squeeze(mean(upd_z_ens(1, :, :)));
mse_upd_x = mean((mean_upd_x - obs_x(2:end)).^2);
corr_upd_x = corr(mean_upd_x, obs_x(2:end));

mean_upd_y = squeeze(mean(upd_z_ens(2, :, :)));
mse_upd_y = mean((mean_upd_y - obs_y(2:end)).^2);
corr_upd_y = corr(mean_upd_y, obs_y(2:end));

%% Plot results of ensemble Kalman filtering
fig = figure;
fig.Color = [1,1,1];
sgtitle(['Results of Ensemble Kalman Filtering', newline, ...
    'for the Prey-Predator Model'], 'FontWeight', 'bold')

% Prey
subplot(2,1,1)
for j = 1:num_ensembles
    plot(obs_t(j+1)*ones(N, 1), upd_z_ens(1, :, j), 'k.', 'MarkerSize', 2)
    hold on
    plot(obs_t(j+1), m_hat(j, 1), 'ro', 'LineWidth', 2)
    plot(obs_t(j+1), obs_x(j+1), 'bx', 'LineWidth', 2)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlabel('Time')
    ylabel('Population')
    title('Prey')
    xlim([0 370])
    ylim('padded')
    legend('Updated Ensembles', 'Predicted Means', 'Measured Values', ...
        'Location','northwest')
end

% Predator
subplot(2,1,2)
for j = 1:num_ensembles
    plot(obs_t(j+1)*ones(N, 1), upd_z_ens(2, :, j), 'k.', 'MarkerSize', 2)
    hold on
    plot(obs_t(j+1), m_hat(j, 2), 'ro', 'LineWidth', 2)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlabel('Time')
    ylabel('Population')
    title('Predator')
    xlim([0 370])
    ylim('padded')
    legend('Updated Ensembles', 'Predicted Means', ...
        'Location','northwest')
end