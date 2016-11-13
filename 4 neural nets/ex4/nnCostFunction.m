function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%PART=1
% for each example i we are going to add the cost for every k=1 to 10

 % Add ones or bias unit to the X data matrix
 X = [ones(m, 1) X];
 z2 = X * Theta1' ;
 
 a_secoundLayer = sigmoid(z2) ; % a2 = g(z1) 
 
 % 5000x25 adding bias unit of second layer to make 5000x26
 
  a_secoundLayer = [ ones(size(a_secoundLayer,1),1) , a_secoundLayer ] ;
 
  z3 = a_secoundLayer * Theta2' ; 
  
  h = sigmoid(z3) ; % 5000x10 hypothesis h_theta(x)
  
  % now in a_3 each column is giving the probability of getting a fixed number
  % column 1 for 1 , 9 for 9 and column 10 for 0
  % each training example is in a row where probability of getting a numbers is given
  % this is called h(x) and we will se the deviation of it from y ie. the answer
  
  % in neural network as compared to logistic regression has multiple layers
  % else in logistic regression we could have just applied it for input and 
  % output layer. there is no layer up in middle.
  %we used one vs all method to compute the possibility of how strong a given
  %input out of the rest all possible input is to the answer
  
  J=0;
  
  for i=1:m 
  
  %rewriting y as vectors of zeroes with one at the right answer place.  
  y_vect = zeros(num_labels,1);
  y_vect(y(i))=1; % size 10x1
  %size h 1x10
  sum = 0;
      
    for k=1:num_labels
    
    J_SingleExample = ( -log(h(i,:))*y_vect - log(1-h(i,:))*(1-y_vect) ) ;
    sum += J_SingleExample;
    
    endfor ;
     J += J_SingleExample;
    
    endfor ;

    J = J/m;
 %theta 1 = each row is for one training example and each column is for the 
 % output presented by previous layer's one fixed node.
 
 %implementing regularisation
 % squaring and averaging all theta values for one training example with 
 % the magnitude of effect using lambda values inculded 
 % and adding to cost function prevents overfitting etc etc
 %also here average of all the training example is also taken for
%  final cost calculation that is why 2 sum func is used
 Theta1_sum = 0;
 Theta2_sum = 0 ;
 
  for i=1:size(Theta1,1)
      for j= 2:size(Theta1,2) % J=2 for neglecting bias unit column number 1
       Theta1_sum += (Theta1(i,j))^2;
      
      endfor;
   endfor ; 

   for i=1:size(Theta2,1)
      for j= 2:size(Theta2,2) % J=2 for neglecting bias unit column number 1
       Theta2_sum += (Theta2(i,j))^2;
      
      endfor;
   endfor ; 

 regressionValue =  (lambda/(2*m))*(Theta1_sum + Theta2_sum) ;

 J = J + regressionValue ; % FINAL COST FUNCTION
% -------------------------------------------------------------

% PART 2 - FINDING THE GRADIENTS USING BACKPROPAGATION ALGO
grad1=zeros(size(Theta1));
grad2=zeros(size(Theta2));

 for i=1:m %for taking one training example at a time

% Added before ones or bias unit to the X data matrix.

 example = X(i,:); % taking one example at a time 1x401
 
  z2 = example * Theta1' ; % 1x401 X 401x25 = 1x25
  
   a_secoundLayer = sigmoid(z2) ; % a2 = g(z1) 
 
 % 1x25 adding bias unit of second layer to make 1x26
 
  a_secoundLayer = [ ones(size(a_secoundLayer,1),1) , a_secoundLayer ] ;
 
  z3 = a_secoundLayer * Theta2' ; % 1x26 X 26x10 
  
  h = sigmoid(z3) ; % 1x10 
 
  h=h'; %10x1 hypothesis h_theta(x) Our output
  
  %Computing the error
  
  %rewriting y as vectors of zeroes with one at the right answer place.  
  y_vect = zeros(num_labels,1);
  y_vect(y(i))=1; % size 10x1
  
  delta3 = h - y_vect ; % size 10x1
  
  %using backpropagation
  
  g1= sigmoidGradient(z2');
  g1=[1;g1]; %adding bias term to make it 26x1
  
  delta2 =((Theta2')*delta3).*g1; %26x10 X 10x1 = 26x1 .*multiplies elementwise

  %delta1=0 as there cannot be any error in inputs given..
  
   % removing bias term for gradient calculation in delta2
   % delta3 = delta3 only as the output layer does not have bias term
   
   grad2 = grad2 + delta3*a_secoundLayer; % 10x1 X 1x26 = 10x26
   grad1 = grad1 + delta2(2:end)*example;  %  25x1 1x401 = 25x401 

 endfor;

%taking the average or the accumulated gradient =

 regularisingTerm =  (lambda/m).*Theta1;
 Theta1_grad = grad1./m + regularisingTerm ;
 
 % removing changes in bias term, coz that is not regularised
 Theta1_grad(:,1)= Theta1_grad(:,1) - (lambda/m).*Theta1(:,1);
 
 
 regularisingTerm2 =  (lambda/m).*Theta2;
 Theta2_grad = grad2./m + regularisingTerm2 ;
 Theta2_grad(:,1)= Theta2_grad(:,1) - (lambda/m).*Theta2(:,1);
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
