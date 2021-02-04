%% Run a contaminated set of analysis to determine the success of NN training 
% for number 1:5 identification
% Includes dropout
%%
clear all

TestDeepDropout; % This trains the W1/W2 weights.
X = zeros(5, 5, 5);

% Load non-ideal number "pixel matrices".
X(:, :, 1) = [ 
0 0 1 1 0;
0 0 1 1 0;
0 1 0 1 0;
0 0 0 1 0;
0 1 1 1 0
];


X(:, :, 2) = [
1 1 1 1 0;
0 0 0 0 1;
0 1 1 1 0;
1 0 0 0 1;
1 1 1 1 1
];


X(:, :, 3) = [ 
1 1 1 1 0;
0 0 0 0 1;
0 1 1 1 0;
1 0 0 0 1;
1 1 1 1 0
];


X(:, :, 4) = [ 
0 1 1 1 0;
0 1 0 0 0;
0 1 1 1 0;
0 0 0 1 0;
0 1 1 1 0
];


X(:, :, 5) = [ 
0 1 1 1 1;
0 1 0 0 0;
0 1 1 1 0;
0 0 0 1 0;
1 1 1 1 0
];

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