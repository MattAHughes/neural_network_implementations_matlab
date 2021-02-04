%% Train the deep NN for number classification of simple 5 x 5 image matrices
% Implement drop out in this 
% Uses sigmoid function, may get better results (if the training set was of
% real, non-dealised data, with ReLU.
%%
clear all

X = zeros(5, 5, 5);

% A perfect 1
X(:, :, 1) = [
0 1 1 0 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 1 0 0;
0 1 1 1 0
];

% A perfect 2
X(:, :, 2) = [ 
1 1 1 1 0;
0 0 0 0 1;
0 1 1 1 0;
1 0 0 0 0;
1 1 1 1 1
];

% A perfect 3
X(:, :, 3) = [ 
1 1 1 1 0;
0 0 0 0 1;
0 1 1 1 0;
0 0 0 0 1;
1 1 1 1 0
];

% A perfect 4
X(:, :, 4) = [
0 0 0 1 0;
0 0 1 1 0;
0 1 0 1 0;
1 1 1 1 1;
0 0 0 1 0
];

% A perfect 5
X(:, :, 5) = [
1 1 1 1 1;
1 0 0 0 0;
1 1 1 1 0;
0 0 0 0 1;
1 1 1 1 0
];

% One hot output row vectors of 1 to 5
D = [ 1 0 0 0 0;
0 1 0 0 0;
0 0 1 0 0;
0 0 0 1 0;
0 0 0 0 1
];

% Initialize the weights
W1 = 2*rand(20, 25) - 1;
W2 = 2*rand(20, 20) - 1;
W3 = 2*rand(20, 20) - 1;
W4 = 2*rand( 5, 20) - 1;

for epoch = 1:20000 % training
[W1, W2, W3, W4] = DeepDropout(W1, W2, W3, W4, X, D);
end

N = 5; % inference
for k = 1:N
x = reshape(X(:, :, k), 25, 1);
v_1 = W1*x;
y_1 = Sigmoid(v_1);
v_2 = W2*y_1;
y_2 = Sigmoid(v_2);
v_3 = W3*y_2;
y_3 = Sigmoid(v_3);
v = W4*y_3;
y = Softmax(v)
end