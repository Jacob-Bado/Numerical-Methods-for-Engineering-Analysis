% MCMC Sampling Algorithm, Jacob Bado, 3-18-24

function [chains, acc_prob, acc_rate] = mcmc_sampling(x0, delta, f, N)

% Obtain number of chains
n_chains = length(x0);

% Preallocate memory
acc_prob = zeros(N, n_chains);
acc_rate = zeros(1, n_chains);
chains = zeros(N, n_chains);

% Set the first element of each chain to be x0
chains = [x0; chains]; 

% Perform sampling for each chain
for j = 1:n_chains
    % Track acceptance
    acc_count = 0;
    
    % Compute Markov chains
    for k = 1:N
        % Compute proposal and current values
        prop_val = unifrnd(chains(k, j) - delta/2, chains(k, j) + delta/2);
        curr_val = chains(k, j);

        % Compute acceptance probability
        acc_prob(k, j) = min(1, f(prop_val)/f(curr_val));
    
        % Accept or reject proposal value
        if acc_prob(k, j) > unifrnd(0, 1)
            chains(k+1, j) = prop_val;
            acc_count = acc_count + 1;

        % Accept current value
        else
            chains(k+1, j) = curr_val;
        end
    end

    % Compute acceptance rate
    acc_rate(j) = acc_count/N;
end