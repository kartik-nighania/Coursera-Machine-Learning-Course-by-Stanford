function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Kartik implementation 
% we will have to add a column of 1 to x coz theta has n+1 rows or
%inshort the x0 element theta0 where x0 is always 1

 % X = [ones(size(X),1),X]; already done in main file
 h = 1./(1 + e.^(-( X*theta )));  % our prediction = m x 1 dimension
 
 %J is finally one number that is the average of all the error found in 
 %each x y training set example for a given value of theta 
 %u could either use'.' elementary multiplication function or calculate 
 %dimensions smartly so that sum can happen in matrix multiplication 
 %only using transpose.
 J = (1/m) * ( -log(h)'*y - log(1-h)'*(1-y) );
 
 % by transpose only that particular element for example x0 with theta0
 % x1 with theta1 gets multiplied as required in Grad Descent
 grad = (1/m)*((X')*(h-y));
 
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
