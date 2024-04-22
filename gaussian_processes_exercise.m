% A simple exercise for understanding gaussian processes and how they can
% be used to recover probability distributions from data
% Jacob Bado, 4-2-24

% (a) Generate realizations of a gaussian process with different parameters

% Define input space for the Gaussian process
x = -5:0.1:5;

% Define vector of length scales
l = [0.2, 1, 5];

% Define mean of Gaussian process
mu = zeros(length(x), 1);

% Preallocate memory
k = zeros(length(x), length(x), length(l));

% Compute squared exponential covariance matrix
for p = 1:length(l)
    k(:, :, p) = exp(-0.5*((x - x')/l(p)).^2);
end

% Compute realizations of the GP
reals_4a = zeros(length(l), length(x));
for i = 1:length(l)
    reals_4a(i, :) = mvnrnd(mu, k(:,:, i), 1);
end

% Plot realizations
fig = figure; hold on
fig.Color = [1, 1, 1];
plot(x, reals_4a(1, :), 'LineWidth', 1.5, 'Color', [0.5, 0.1, 0.1])
plot(x, reals_4a(2, :), 'LineWidth', 1.5, 'Color', [0, 0.6, 0.6])
plot(x, reals_4a(3, :), 'LineWidth', 1.5, 'Color', [0.4, 0, 0.7])

ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
box on
xlim('tight')
ylim('padded')
xlabel('Input Space')
ylabel('Output Space')
title('Gaussian Processes')
legend('l = 0.2', 'l = 1', 'l = 5', 'Location', 'best')

%% (b) Generate 20 realizations of a gaussian process with fixed parameters

% Define number of realizations to generate
n_reals = 20;

% Preallocate memory
reals_4b = zeros(n_reals, length(x));

% Generate 20 realizations for the covariance matrix with l = 1
for i = 1:n_reals
    reals_4b(i, :) = mvnrnd(mu, k(:,:, 2), 1);
end

% Plot realizations
fig = figure;
fig.Color = [1, 1, 1];
sgtitle('Gaussian Processes, l = 1', 'FontWeight', 'bold')
for i = 1:n_reals
    subplot(2, 2, 1)
    plot(x, reals_4b(i, :), 'k', 'LineWidth', 1.5); hold on
end

ax = gca;
ax.LineWidth = 1.5;
ax.FontWeight = 'bold';
xlim('tight')
ylim('padded')
xlabel('Input Space')
ylabel('Output Space')
title('Prior Realizations')

%% (c) Use a selected gaussian process to generate data, then obtain a
% posterior distributions for subsets of the data 

% Select a realization
real_4c = reals_4b(randi(n_reals), :);

% Highlight the realization that was selected in the subplot
subplot(2, 2, 1)
plot(x, real_4c, 'r', 'LineWidth', 3); hold on

% Define input spaces
x_i = {-5:1:5, -5:2:5, -5:5:5};

% Compute posteriors and plot realizations using each of the 3 datasets
for i = 1:length(x_i)
    % Obtain data from fixed realization
    f = real_4c(ismember(x, x_i{i}));
    
    % Compute cross-covariance matrix
    k_cross = exp(-0.5*((x - x_i{i}')/1).^2);

    % Compute covariance matrix of smaller dataset
    k_data = exp(-0.5*((x_i{i} - x_i{i}')/1).^2);

    % Obtain posterior mean 
    mu_post = k_cross'*(k_data\f');
    
    % Obtain posterior covariance
    k_post = k(:, :, 2) - k_cross'*(k_data\k_cross);

    % Enforce posterior covariance symmetry due to numerical rounding error
    k_post = (k_post + k_post')./2;

    % Generate realizations from posterior distributions
    for j = 1:n_reals
        reals_4c = mvnrnd(mu_post, k_post);

        % Plot realizations
        subplot(2, 2, i+1)
        plot(x, reals_4c, 'k', 'LineWidth', 1.5); hold on

        ax = gca;
        ax.LineWidth = 1.5;
        ax.FontWeight = 'bold';
        xlim('tight')
        ylim('padded')
        xlabel('Input Space')
        ylabel('Output Space')
        title(['Posterior Realizations, n = ', num2str(length(x_i{i}))])
    end
end