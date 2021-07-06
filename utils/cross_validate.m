function score = cross_validate(y, X, algorithm, perf_metr, n_folds)
% This code is based on that of cross validation from ITML
%(Information theoretic Metric learning)


[n, ~] = size(X);
if (n ~= length(y))
    disp('ERROR: num rows of X must equal length of y');
    return;
end

% Permute the rows of X and y
s = RandStream('dsfmt19937');
rp = randperm(s, n);
y = y(rp);
X = X(rp,:);

% Initializing different measure
scores = zeros(n_folds, 1);

for i=1:n_folds
    
    %% splitting the data to test and train
    test_start = ceil(n/n_folds * (i-1)) + 1;
    test_end = ceil(n/n_folds * i);
    
    y_train = [];
    X_train = [];
    if i > 1
        y_train = y(1:test_start-1);
        X_train = X(1:test_start-1,:);
    end
    if i < n_folds
        y_train = [y_train; y(test_end+1:length(y))];
        X_train = [X_train; X(test_end+1:length(y), :)];
    end
    
    X_test = X(test_start:test_end, :);
    y_test = y(test_start:test_end);
    
    %% learning the divergence and clustering/knn/ranking with it
    bregman_div = algorithm(y_train, X_train);
    scores(i) = perf_metr(y_train, X_train, y_test, X_test, bregman_div);
    
end
score = mean(scores);