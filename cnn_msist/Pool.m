%% A mean pooling function

function y = Pool(x)
% 2x2 mean pooling

[x_rows, x_cols, num_filters] = size(x);
y = zeros(x_rows / 2, x_cols / 2, num_filters);

for filter_num = 1:num_filters
filter = ones(2) / (2*2); % for mean
image = conv2(x(:, :, filter_num), filter, 'valid');
y(:, :, filter_num) = image(1:2:end, 1:2:end);
end
end