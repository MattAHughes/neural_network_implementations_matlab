% Description - The sigmoid function
% Comment - Pairs with DeltaSGD.m and TestDeltaSGD.m
% Parameters - x = a single input
%            - y = sigmoid output
%%
function y = Sigmoid(x)
y = 1 ./ (1 + exp(-x));
end