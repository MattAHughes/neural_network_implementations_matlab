%% The softmax function.

function y = Softmax(x)
exp_x = exp(x);
y = exp_x / sum(exp_x);
end