% This function implements the Steepest Descent algorithm for a 
% predetermined objective function, as part of another exercise.
% Jacob Bado, 3-16-24

function [x, l2x_x1, fx, l2dfx, l2x_xmin] = ...
    steepest_descent(x0, x_min, err_tol)

    % Define the objective function
    f = @(x, y, z) x.^2 - x.*y + 2*y.^2 + x.*z + z.^2 + y.*z + 6*z;

    % Define the gradient
    df = @(x, y, z) [(2*x - y + z); (-x + 4*y + z); (x + y +2*z + 6)];

    % Define the Hessian
    d2f = [2, -1, 1;  -1, 4, 1;  1, 1, 2];

    % Preallocate variables
    x = x0;
    k = 1;
    l2dfx = vecnorm(df(x(1,:), x(2,:), x(3,:)));

    % Iterate using steepest descent algorithm
    while l2dfx(k) > err_tol
        % Compute step size
        a = (df(x(1,k), x(2,k), x(3,k))' * df(x(1,k), x(2,k), x(3,k))) /...
          (df(x(1,k), x(2,k), x(3,k))' * d2f * df(x(1,k), x(2,k), x(3,k)));
        
        % Perform steepest descent iteration
        x1 = x(:,k) - a*df(x(1,k), x(2,k), x(3,k));
        
        % Store x,y,z iteration points
        x = [x, x1];

        % Compute the norm of the gradient vector at each iteration point
        l2dfx = [l2dfx, vecnorm(df(x(1,k), x(2,k), x(3,k)))];

        % Add to counter k for each iteration
        k = k+1;
    end
    
    % Compute the distance between two consecutive iteration points
    l2x_x1 = vecnorm(diff(x, [], 2));

    % Compute the objective function's value at each iteration point
    fx = f(x(1,:), x(2,:), x(3,:));

    % Compute distance between each iteration point and the minimizer
    l2x_xmin = vecnorm(x - x_min);
end
