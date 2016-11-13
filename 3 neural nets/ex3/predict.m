function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% adding the bias unit to input.
 X = [ ones(m,1) , X ] ;

% Also parameters are already trained and 
% set with the input and now the inputs will get us the output 
% no for loops is used and we will get the prediction value for each label 
%  1- 10 (10 for 0) for each sample. 
%also sigmod has to be implemented to form a of the next layer
%the first multipication gives us 5000x25. 5000 no of examples and 25 values
%where Each Value is the product sum obtained that is called z and now to make it
% a2 we will have to use the sigmoid and make a^2 = g(z). for further process.

 z2 = X * Theta1' ;
 
 a_secoundLayer = 1.0 ./ (1.0 + exp(-z2)) ; % a2 = g(z1) 
 
 % 5000x25 adding bias unit of second layer to make 5000x26
 
  a_secoundLayer = [ ones(size(a_secoundLayer,1),1) , a_secoundLayer ] ;
 
  z3 = a_secoundLayer * Theta2' ; 
  
  a_3 = 1.0 ./ (1.0 + exp(-z3)) ; % 5000x10
  
  % now in a_3 each column is giving the probability of getting a fixed number
  % column 1 for 1 , 9 for 9 and column 10 for 0
  % each training example is in a row where probability of getting a numbers is given
  % the bestpossible output number is the max value in a row, and the index of that column
  % is the predicted number
  
   [values,index] = max(a_3'); 
 %2nd argument returns the index value 
 %what we wanted =  row wise of the max value of probability predicted.
 
  p =index';
% =========================================================================

end
