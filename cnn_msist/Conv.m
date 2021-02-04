%% The convolution function for MnistConv

function y = Conv(x, W)

[w_rows, w_cols, num_filters] = size(W);
[x_rows, x_cols, ~ ] = size(x);

y_row = x_rows - w_rows + 1;
y_col = x_cols - w_cols + 1;

y = zeros(y_row, y_col, num_filters);

for filter_num = 1:num_filters
    filter = W(:, :, filter_num);
    filter = rot90(squeeze(filter), 2);
    y(:, :, filter_num) = conv2(x, filter, 'valid');
end

end