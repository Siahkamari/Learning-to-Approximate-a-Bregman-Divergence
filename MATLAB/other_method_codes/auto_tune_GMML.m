function [bregman_div, params] = auto_tune_GMML(y, X, m, task)


lambdas = 0.1;
t = 0:0.2:1;
[LAMBDAS, T] = meshgrid(lambdas, t);
LAMBDAS = LAMBDAS(:);
T = T(:);
n_folds = 3;

scores = zeros(length(LAMBDAS), 1);
for i=1:length(LAMBDAS)
    fprintf('\tTuning GMML: lambda = %.4f, K = %d \n', LAMBDAS(i), T(i));
    
    algorithm = @(y,X) GMML(y, X, m, LAMBDAS(i), T(i));
    perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
    scores(i) = cross_validate(y, X, algorithm, perf_metr, n_folds);
end

[~,i] = max(scores);
lambda = LAMBDAS(i);
t = T(i);

% n_hplane = 100;
% lambda = 1e-2;

fprintf('Optimal lambda : %.4f, K: %d \n', lambda, t);

[bregman_div, params] = GMML(y, X, m, lambda, t);
