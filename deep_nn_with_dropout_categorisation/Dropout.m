%% Define how we determine dropout

function ym = Dropout(y, ratio)

[m, n] = size(y);
ym = zeros(m, n);

dropout_num = round(m * n * (1 - ratio));
idx = randperm(m * n, dropout_num);

ym(idx) = 1 / (1 - ratio);
end