% Imlement an acceptance-rejection algorithm to generate samples from a
% beta distribution.
% Jacob Bado, 3-11-24

% Define Beta distribution parameters
a = 2;
b = 5;

% Define ranges of x values
x_vec = 0:0.001:1;

% Generate beta distribution f(x)
f = betapdf(x_vec, a, b);

% Find max density x location for f(x), this will be the mean of g(x)
g_mean = x_vec(f == max(f));

% Define g(x) standard deviation
g_std = 0.22;

% Generate gaussian distribution g(x)
g = normpdf(x_vec, g_mean, g_std);

% Programmatically determine the appropriate scaling constant M
M = max(f)/max(g);

% Show that f(x) <= M*g(x)
valid_check = min(M*g - f);

% Define sample size exponent
n = 1:6;

% Preallocate memory
accepted_samps = cell(max(n), 1);
accept_rate = zeros(1, sum([10^1, 10^2, 10^3, 10^4, 10^5, 10^6]));
counter = 0;

for i = n
    % Define sample size
    N = 10^i;
    
    % Define variables to track acceptance
    accepted_samps{i} = zeros(1, N);
    n_accepted = 0;
    
    % Iterate until N samples have been accepted
    while n_accepted < N
        counter = counter + 1;
        % Generate random sample from g(x)
        x = normrnd(g_mean, g_std, 1);
    
        % Generate uniform random number from [0, 1]
        u = unifrnd(0, 1, 1);
    
        % Compute acceptance probability
        fx = betapdf(x, a, b);
        gx = normpdf(x, g_mean, g_std);
        rho =  fx/(M*gx);
    
        % Accept or reject sample
        if u <= rho
            n_accepted = n_accepted + 1;
            accepted_samps{i}(n_accepted) = x;
            accept_rate(counter) = 1;
        end
    end
end

% Compute acceptance rate
acceptance_rate = 100*sum(accept_rate)/length(accept_rate);

%% Plot target and proposed distributions

gx_vec = linspace(-4*g_std+g_mean, 4*g_std+g_mean, length(x_vec));

fig = figure;
fig.Color = [1,1,1];
plot(x_vec, f, ...
    'Color', [0.6, 0, 0.2], 'LineWidth', 2)
xlim('tight')
hold on
plot(gx_vec, M*normpdf(gx_vec, g_mean, g_std), '--', ...
    'Color', [0, 0.6, 0.4], 'LineWidth', 2)

ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';

xlabel('x')
ylabel('Probability Density')
title('Target and Proposed Distributions')
legend('f(x)', 'M*g(x)')

%% Plot target distribution and histograms of accepted samples

bin_edges = linspace(0, 1, 30);

fig = figure;
fig.Color = [1,1,1];
sgtitle('Comparison of Sample Histograms and Target Distribution')

for i = n
    subplot(3,2,i)
    hi = histogram(accepted_samps{i}, bin_edges, 'Normalization', 'pdf', ...
        'FaceColor', [0, 0.6, 0.4], 'LineWidth', 1);
    hold on
    plot(x_vec, f, ...
        'Color', [0.6, 0, 0.2], 'LineWidth', 2)
    xlim('tight')

    ax = gca;
    ax.LineWidth = 2;
    ax.FontWeight = 'bold';
    
    xlabel('x')
    ylabel('Probability Density')
    title(['N Samples: ', num2str(10^i)])

    if i == 6
        legend('Sample Histogram', 'Target Distribution', ...
            'Location', 'northeast')
    end
end