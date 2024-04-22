% Ensemble Kalman Filter for Prey-Predator Model.
% Jacob Bado, 3-17-24

function [m_hat, C_hat, upd_z_ens] = ensemble_kf_preypred...
    (z0_ens, num_ensembles, N, s, Q, R, H, obs_t, obs_x, par)
    
    % Preallocate memory
    pre_z_ens = zeros(N, 2);
    m_hat = zeros(num_ensembles, 2);
    C_hat = zeros(2, 2, num_ensembles);
    upd_z_ens = zeros(2, N, num_ensembles);
    
    % Define prey-predator dynamical system
    z_prime = @(t, z) [par(1,1)*z(1) - par(1,2)*z(1)*z(2); ...
                       par(2,1)*z(1)*z(2) - par(2,2)*z(2)];
    
    for j = 1:num_ensembles
    %%%%%%%%%% Prediction %%%%%%%%%%
        
        % Define time span for dynamical system prediction
        t_span = [obs_t(j), obs_t(j+1)];
        
        % Generate prediction ensemble by solving the dynamical system
        if j == 1 % If measurement time is zero, use initial ensemble
            for i = 1:N
                [~, pre_z] = ode45(z_prime, t_span, z0_ens(:, i));
                pre_z_ens(i, :) = pre_z(end, :);
            end
        else % If measurement time is nonzero, use updated ensemble
            for i = 1:N
                [~, pre_z] = ode45(z_prime, t_span, upd_z_ens(:, i, j-1));
                pre_z_ens(i, :) = pre_z(end, :); 
            end
        end
        xi_ens = mvnrnd([0; 0], Q, N);
        pre_z_ens = pre_z_ens + xi_ens;
       
        % Compute empirical mean of prediction ensemble
        m_hat(j, :) = sum(pre_z_ens)/N;
        
        % Compute empirical covariance of prediction ensemble
        C_hat(:, :, j) = (pre_z_ens - m_hat(j, :))'* ...
                         (pre_z_ens - m_hat(j, :))/N;
        
    %%%%%%%%%% Innovation %%%%%%%%%%
    
        % Compute innovation variance
        S = H*C_hat(:, :, j)*H' + R;
        
        % Compute Kalman gain
        K = C_hat(:, :, j)*H'*S^-1;
        
    %%%%%%%%%% Analysis %%%%%%%%%%
    
        % Generate observation ensemble
        eta_ens = mvnrnd(0, R, N);
        obs_x_ens = obs_x(j+1)*ones(N, 1) + s*eta_ens;
        
        % Update prey predictions
        upd_z_ens(:, :, j) = ([1, 0; 0, 1] - K*H)*pre_z_ens' + K*obs_x_ens';
    end
end