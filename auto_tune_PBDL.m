function [bregman_div, params] = auto_tune_PBDL(y, X, m, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params);
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy.
%
% Returns: a learned Bregaman divergence


% define lambda the multiplier of the regularization


lambdas = 10.^(-3:-1);
n_hplane = 40:20:120;

[LAMBDAS, N_HPLANE] = meshgrid(lambdas, n_hplane);

knn_size = 5;
n_folds = 3;

out = cell(length(lambdas), 1);
objective = zeros(length(lambdas), 1);
for i=1:length(LAMBDAS(:))
    fprintf('\tTuning PBDL: lambda = %.4f, K = %d \n', LAMBDAS(i), N_HPLANE(i));
    out{i} = cross_validate(y, X, @(y,X) PBDL(y, X, m, LAMBDAS(i), N_HPLANE(i)), n_folds, knn_size, task);
    objective(i) = out{i}{1};
end
%
[~,i] = max(objective);
lambda = LAMBDAS(i);
n_hplane = N_HPLANE(i);

% lambda = 0.01;
% n_hplane = 80;

fprintf('\tOptimal lambda, K value: %.4f, %d \n', lambda, n_hplane);

[bregman_div, params] = PBDL(y, X, m, lambda, n_hplane);
