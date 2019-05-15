function [bregman_div, params] = auto_tune_NBDL(y, X, m, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params);
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy.
%
% Returns: a learned Bregaman divergence



% define lambda the multiplier of the regularization
lambdas = 10.^(-4:-1);
n_hplane = 20:20:140;

[LAMBDAS, N_HPLANE] = meshgrid(lambdas, n_hplane);

knn_size = 5;
n_folds = 3;

out = cell(length(lambdas), 1);
objective = zeros(length(lambdas), 1);
for i=1:length(LAMBDAS(:))
    fprintf('\tTuning NBDL: lambda = %.4f, K = %d \n', LAMBDAS(i), N_HPLANE(i));
    out{i} = cross_validate(y, X, @(y,X) NBDL(y, X, m, LAMBDAS(i), N_HPLANE(i)), n_folds, knn_size, task);
    objective(i) = out{i}{1};
end
%
[~,i] = max(objective);
lambda = LAMBDAS(i);
n_hplane = N_HPLANE(i);
% lambda = 0.001;
fprintf('\tOptimal lambda, K value: %.4f, %d \n', lambda, n_hplane);
for attpets = 1:5
    [bregman_div, params, exitflag] = NBDL(y, X, m, lambda, n_hplane);
    if exitflag
        break
    end
end