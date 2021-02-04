%% Convolutional Neural Network applied to digit (0-9) identification from
%  MNIST database.

%% Author - Vitamin - C
% Inputs - 784 (for 28 x 28 pixel images)
% Feature Extraction CNN - 1 layer, 20 9 x 9 conv. filters
% Hidden Layer - ReLU
% Pooling Layer - 1, Mean pooling 2 x 2 submatrices
% Feeds to - Classification NN, 1 Hidden Layer (100 nodes, ReLU)
% Output - 10 Nodes, Softmax
%%
function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)

alpha = 0.01;
beta = 0.95;

momentum_1 = zeros(size(W1));
momentum_5 = zeros(size(W5));
momentum_o = zeros(size(Wo));

num_numbers = length(D);

batch_size = 100;
batch_start_list = 1:batch_size:(num_numbers - batch_size + 1);

% One epoch loop
for batch = 1:length(batch_start_list)
dW1 = zeros(size(W1));
dW5 = zeros(size(W5));
dWo = zeros(size(Wo));

% Mini-batch loop
begin = batch_start_list(batch);
for k = begin:begin+batch_size-1

    % Forward pass = inference
    x = X(:, :, k); % Input is 28 x 28
    
    y_1 = Conv(x, W1); % Convolution is 20 x 20 x 20
    y_2 = ReLU(y_1); 
    y_3 = Pool(y_2); % Pool, 10x10x20
    y_4 = reshape(y_3, [], 1); 
    
    v_5 = W5 * y_4; % ReLU, 360
    y_5 = ReLU(v_5); 
    
    v = Wo * y_5; % Softmax, 10
    y = Softmax(v); 

    % One-hot encoding
    true_values = zeros(10, 1);
    true_values(sub2ind(size(true_values), D(k), 1)) = 1;

    % Backpropagation
    error = true_values - y; % Output layer
    delta = error;

    error_5 = Wo' * delta; % Hidden(ReLU) layer
    delta_5 = (y_5 > 0) .* error_5;

    error_4 = W5' * delta_5; % Pooling layer

    error_3 = reshape(error_4, size(y_3));

    error_2 = zeros(size(y_2));
    W3 = ones(size(y_2)) / (2*2);

    for c = 1:20
    error_2(:, :, c) = kron(error_3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta_2 = (y_2 > 0) .* error_2; % ReLU layer
    delta_1_x = zeros(size(W1)); % Convolutional layer

    for c = 1:20
    delta_1_x(:, :, c) = conv2(x(:, :), rot90(delta_2(:, :, c), 2), 'valid');
    end
    
    dW1 = dW1 + delta_1_x;
    dW5 = dW5 + delta_5 * y_4';
    dWo = dWo + delta * y_5';
end

% Update weights
dW1 = dW1 / batch_size;
dW5 = dW5 / batch_size;
dWo = dWo / batch_size;

momentum_1 = alpha*dW1 + beta*momentum_1;
W1 = W1 + momentum_1;

momentum_5 = alpha*dW5 + beta*momentum_5;
W5 = W5 + momentum_5;

momentum_o = alpha*dWo + beta*momentum_o;
Wo = Wo + momentum_o;

end
end