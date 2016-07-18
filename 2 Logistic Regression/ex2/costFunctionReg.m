function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

 h = 1./(1 + e.^(-( X*theta ))); 
 
 regression = (lambda/(2*m))*( sum(theta.^2) - (theta(1))^2 ); % theta0 not included
 J = (1/m) * ( -log(h)'*y - log(1-h)'*(1-y) ) + regression ;
 
 grad = (1/m)*((X')*(h-y)) + (lambda/m)*theta;
 grad(1) = grad(1) - (lambda/m)*theta(1) ; % making theta0 unaffected by regression term 
 % indexing in octave and matlab starts from 1 no 0
 
 
% =============================================================

end
