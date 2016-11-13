function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
for c = 1:num_labels

%setting result = 1 and 0 (.ie y) according to the output obtained for label c
%we want to apply logistic regression for single class so using one vs all methods

%result = (y==c); %y=1 for all label values = c else 0.

% now we have output in 0 and 1 for case c (the number). Fit for sigmoid function
% logistic regression. below we are computing parameters for one number.


% To find using GRADIENT DESCENT ALGO
% [cost , grad] = lrCostFunction( zeros((n+1), 1 ) , X , y==c , lambda );
% add our answer grad obtained in our theta set 
% all_theta(c,:) = grad' ;


%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
%     % Set Initial theta
   initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
   options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
     [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                 initial_theta, options);
                 
    all_theta(c,:) = theta'; 

endfor

end
