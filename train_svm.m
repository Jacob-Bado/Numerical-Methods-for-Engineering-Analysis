% Support Vector Classification - Model Training, Jacob Bado, 4-9-24

function [w, b, curr_classes] = train_svm(x_train, y_train, lambda)

% Obtain classes and number of classes
classes = unique(y_train);
k = length(classes);

% Preallocate memory
curr_classes = zeros(k*(k-1)/2, 2);
w = zeros(k*(k-1)/2, width(x_train));
b = zeros(k*(k-1)/2, 1);
counter = 0;

% Loop for k*(k-1)/2 classes
for i = 1:k-1
    for j = i+1:k
        % Increment counter
        counter = counter + 1;

        % Extract x and y data for current classes
        curr_y1_ind = y_train == classes(i);
        curr_y2_ind = y_train == classes(j);
        curr_y1 = curr_y1_ind(curr_y1_ind);
        curr_y2 = -1*curr_y2_ind(curr_y2_ind);
        curr_x1 = x_train(curr_y1_ind, :);
        curr_x2 = x_train(curr_y2_ind, :);
        
        % Define current ground truth and training set
        y_class = [curr_y1; curr_y2];
        x_class = [curr_x1; curr_x2];

        % Define objective function
        obj_fxn = @(pars) (1/height(x_class))* ...
            sum( max(0, 1-y_class.*(x_class*pars(1:end-1) + ...
            pars(end))) ) + lambda*norm(pars(1:end-1))^2;
        
        % Definite paramater initial conditions
        pars_0 = zeros(width(x_class) + 1, 1);
        
        % Define optimization options
        options = optimoptions('fminunc', 'MaxFunctionEvaluations', 10000);

        % Optimize with quasi-newton algorithm (modify_train as needed)
        opt_pars = fminunc(obj_fxn, pars_0, options);
        
        % Extract parameters
        w(counter, :) = opt_pars(1:end-1);
        b(counter) = opt_pars(end);

        % Track currently trained classes
        curr_classes(counter, :) = [classes(i), classes(j)];
    end
end