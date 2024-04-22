% Use Monte-Carlo Methods to Draw Sample From a Target Distribution, 
% Jacob Bado, 3-10-24

% (a) Decide Probability Distribution to Sample From

% Define function to sample from
f = @(x) (x.^2)./exp(x);

% Define function boundaries
bounds = [-5, 5];

% Define probability distribution
pareto = makedist('GeneralizedPareto', 0, 1, bounds(1));

% (b) Implement Monte Carlo sampling
% Define N samples exponent
n = 1:9;

% Preallocate memory
estimate = zeros(length(n), 1);
elapsed = zeros(length(n), 1);

% Iterate
for i = n
    % Define N samples
    N = 10^i;
    
    % Batch process to avoid "Out of memory" errors
    if N > 1e6
        N_batch = 1e6;
        batches = N/N_batch;
        estimate_batch = zeros(batches, 1);

        for j = 1:batches
            % Randomly sample from pi
            u_batch = random(pareto, N_batch, 1);
            
            % Estimate integral of f over [-5, 5] using MC sampling
            estimate_batch(j) = sum(f(u_batch))/N_batch;
        end
        estimate(i) = sum(estimate_batch)/batches;

    % Compute without batch processing
    else
        % Randomly sample from pi
        u = random(pareto, N, 1);

        % Estimate integral of f over [-5, 5] using MC sampling
        estimate(i) = sum(f(u))/N;
    end
end

%% (c) Plot MC Sampling Results

fig = figure;
fig.Color = [1,1,1];

subplot(3,1,1)
plot(estimate, 'x', ...
    'Color', [0.6,0,0], 'LineWidth', 2, 'MarkerSize', 12)
xlim([min(n)-0.5, max(n)+0.5])
ylim('padded')
xticks(n)
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('Iteration (i)')
ylabel('Estimate Value')
title('Integral Estimate')

subplot(3,1,2)
plot(diff(estimate), '+', ...
    'Color', [0.5,0,0], 'LineWidth', 2, 'MarkerSize', 11)
xlim([min(n)-1, max(n)])
xticks(n(1:end-1))
ylim('padded')
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('Iteration (i to i+1)')
ylabel('\Delta Estimate Value')
title('Difference in Integral Estimate')

subplot(3,1,3)
plot(elapsed, 'o', ...
    'Color', [0,0,0.5], 'LineWidth', 2, 'MarkerSize', 6)
xlim([min(n)-0.5, max(n)+0.5])
xticks(n)
ylim('padded')
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('Iteration (i)')
ylabel('Time (sec)')
title('Elapsed Time')

%% Plot Chosen Probability Distribution

x = -5:0.01:5;

par_pdf = pdf(pareto, x);

fig = figure;
fig.Color = [1,1,1];
subplot(2,1,1)
plot(x, par_pdf, 'k', 'LineWidth', 2)
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('x')
ylabel('Probability Density')
title('Pareto Distribution')

subplot(2,1,2)
plot(x, f(x), 'k', 'LineWidth', 2)
ax = gca;
ax.LineWidth = 2;
ax.FontWeight = 'bold';
xlabel('x')
ylabel('f(x)')
title('f(x) = x^2/e^x')