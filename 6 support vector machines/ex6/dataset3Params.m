function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

cVect =   [0.01;0.03;0.1;0.3;1;3;10;30];
sigmaVect = cVect;
x1 = [1 2 1]; x2 = [0 4 -1];
error1 = zeros(size(cVect,1),size(cVect,1));
% making a vector for error equal to size of Xval   

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.

%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
% well train using each combination of c and lambda and will make a model
%we will give this model to predict for xval values.
%well check for the error 
%out of the loop well select the c and lamda for the lowest error index 
% at last we will predict with y and choose the least error giving one
for i=1:size(cVect,1)
for j=1:size(cVect,1)

model2 = svmTrain(X, y, cVect(i) , @(x1, x2) gaussianKernel(x1, x2, sigmaVect(j)));
 
 predictions = svmPredict(model2, Xval);
 error1(i,j) = mean(double(predictions ~= yval));
 
end
end

[num idx] = min(error1(:));
[x y] = ind2sub(size(error1),idx);
C=cVect(x);
sigma=sigmaVect(y);

return;
% =========================================================================

end
