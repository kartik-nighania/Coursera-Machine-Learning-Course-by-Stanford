function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
sigmoidFunc = 1.0 ./ (1.0 + exp(-z)); % sigmoid gradient is differentiation of 
%gradient function that is g(Z) = g(z)(1-g(z))

g = sigmoidFunc.*(1- sigmoidFunc);
end
