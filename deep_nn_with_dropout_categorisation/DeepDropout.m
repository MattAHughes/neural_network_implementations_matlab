%% Implementation of deep learning with dropout
% NN with 3 hidden layers of 20 nodes. 
% Uses ReLU to avoid vanishing gradients, with cross entropy loss
% Uses softmax output and dropout to avoid overtraining
% Identification of digits 1 to 5 based on 25 pixel array (black or white)
% Inputs - Weights for each layer, inputs X and true outputs D.

function [W1, W2, W3, W4] = DeepDropout(W1, W2, W3, W4, X, D)

alpha = 0.01;
N = 5;

for k = 1:N
    x = reshape(X(:, :, k), 25, 1);
    
    v_1 = W1*x;
    y_1 = Sigmoid(v_1);
    y_1 = y_1 .* Dropout(y_1, 0.2);

    v_2 = W2*y_1;
    y_2 = Sigmoid(v_2);
    y_2 = y_2 .* Dropout(y_2, 0.2);

    v_3 = W3*y_2;
    y_3 = Sigmoid(v_3);
    y_3 = y_3 .* Dropout(y_3, 0.2);

    v = W4*y_3;
    y = Softmax(v);

    d = D(k, :)';
    e = d - y;
    delta = e;

    e_3 = W4'*delta;
    delta_3 = y_3.*(1-y_3).*e_3;
    
    e_2 = W3'*delta_3;
    delta_2 = y_2.*(1-y_2).*e_2;

    e_1 = W2'*delta_2;
    delta_1 = y_1.*(1-y_1).*e_1;

    dW4 = alpha*delta*y_3';
    W4 = W4 + dW4;

    dW3 = alpha*delta_3*y_2';
    W3 = W3 + dW3;

    dW2 = alpha*delta_2*y_1';
    W2 = W2 + dW2;

    dW1 = alpha*delta_1*x';
    W1 = W1 + dW1;
end
end