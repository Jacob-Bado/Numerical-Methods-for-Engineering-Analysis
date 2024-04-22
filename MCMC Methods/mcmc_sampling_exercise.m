% This is an exercise to assist in understanding Markov Chain Monte-Carlo
% Methods and how they can be used to recover a target distribution.
% Jacob Bado, 3-18-24

% Define x0
x0 = [2, 80, -10, 100, -250, -20, 35, 0];

% Define delta
delta = 4;

% Define target distribution function handle
f = @(x) exp((-1/2) * ((x - 5)/3).^2);

% Define number of iterations
N = 10000;

% Call MCMC sampling function
[chains, acc_prob, acc_rate] = mcmc_sampling(x0, delta, f, N);

%% Plot results
fig = figure;
fig.Color = [1,1,1];
sgtitle('Markov Chains', 'FontWeight', 'bold')

for i = 1:length(x0)
    subplot(length(x0), 1, i)
    plot(0:N, chains(:, i), 'k-')
    
    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlim('tight')
    ylim('padded')
    xlabel('Iteration (k)')
    ylabel('Value')
    title(['x_0 = ', num2str(x0(i)), ', \delta = ', num2str(delta)])
end

%% Plot results (shorter window)
fig = figure;
fig.Color = [1,1,1];
sgtitle('Markov Chains', 'FontWeight', 'bold')

for i = 1:length(x0)
    subplot(length(x0), 1, i)
    plot(0:500, chains(1:501, i), 'k-', 'LineWidth', 1.5)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlim('tight')
    ylim('padded')
    xlabel('Iteration (k)')
    ylabel('Value')
    title(['x_0 = ', num2str(x0(i)), ', \delta = ', num2str(delta)])
end

%% Select chains that appear to converge faster, select burn-in period

% Select chain indices for chains that visually appear to converge faster
good_chain_inds = [1, 3, 6, 7, 8];

% Define burn-in period based on visual inspection
burn_in = 250;

% Extract chains that appear to converge faster & remove burn-in
good_chains = chains(burn_in:end, good_chain_inds);

% Compute remaining chain length
L = height(good_chains);

% Compute chain means
chain_mean = mean(good_chains);

% Compute between chain variance
bcv = L*var(chain_mean);

% Compute within chain variances
wcv = var(good_chains);

% Compute within chain variance mean
w = mean(wcv);

% Compute Gelman-Rubin statistic
R = (((L-1)/L)*w + bcv/L)/w;

%% Run MCMC simulation with different delta values
x0 = 2;
deltas = [0.1, 0.5, 1, 2, 4, 8, 16, 32];

% Preallocate memory
acc_prob_2 = zeros(N, length(deltas));
acc_rate_2 = zeros(1, length(deltas));
chains_2 = zeros(N+1, length(deltas));

% Loop for different delta values
for i = 1:length(deltas)
    [chains_2(:, i), acc_prob_2(:, i), acc_rate_2(i)] = ...
        mcmc_sampling(x0, deltas(i), f, N);
end

%% Plot results of second simulation
chains_to_plot = [1, 5, 8];

mu = 5;
sigma = 3;
x_pdf = linspace(mu - 4*sigma, mu + 4*sigma, 100);
auc = trapz(x_pdf, f(x_pdf));
f_norm = f(x_pdf)/auc;

fig = figure;
fig.Color = [1,1,1];
sgtitle('Comparison of Sample Histograms and Target Distribution', ...
    'FontWeight', 'bold')

for i = 1:length(chains_to_plot)    
    bin_edges = linspace(min(x_pdf), max(x_pdf), 80);

    subplot(length(chains_to_plot), 1, i)
    hi = histogram(chains_2(:, chains_to_plot(i)), bin_edges, ...
       'Normalization', 'pdf', 'FaceColor', [0, 0.6, 0.4], 'LineWidth', 1);
    hold on
    plot(x_pdf, f_norm, ...
        'Color', [0.6, 0, 0.2], 'LineWidth', 2)

    ax = gca;
    ax.LineWidth = 1.5;
    ax.FontWeight = 'bold';
    xlabel('Values')
    ylabel('Probability Density')
    xlim('tight')
    ylim([0 max(hi.Values) + 0.1*max(hi.Values)])
    title(['x_0 = ', num2str(x0), ...
        ', \delta = ', num2str(deltas(chains_to_plot(i)))])
end