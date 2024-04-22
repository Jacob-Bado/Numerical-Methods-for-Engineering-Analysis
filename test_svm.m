% Support Vector Classification - Model Testing, Jacob Bado, 4-9-24

function predictions = test_svm(x_test, curr_classes, w, b)

% Preallocate memory
predictions = zeros(height(x_test), 1);

% Loop for each testing sample
for i = 1:height(x_test)
    % Pull out current testing sample
    curr_x_test = x_test(i, :);
    
    % Preallocate memory
    prediction = zeros(height(w), 1);
    
    % Loop for each classifier
    for j = 1:height(w)
        % Classify test data
        class = @(curr_x_test) w(j, :)*curr_x_test' + b(j);
        if sign(class(curr_x_test)) == 1
            prediction(j) = curr_classes(j, 1);
        elseif sign(class(curr_x_test)) == -1
            prediction(j) = curr_classes(j, 2);
        end
    end

    % Obtain prediction as the mode prediction vote
    predictions(i) = mode(prediction);
end