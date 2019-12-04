function [bregman_div, params] = auto_tune_PBDL(y, X, m, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params);
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy.
%
% Returns: a learned Bregaman divergence

% define lambda the multiplier of the regularization

lambdas = 10.^(-3:1);
n_hplane = 40:20:120;
[LAMBDAS, N_HPLANE] = meshgrid(lambdas, n_hplane);
LAMBDAS = LAMBDAS(:);
N_HPLANE = N_HPLANE(:);
n_folds = 3;

scores = zeros(length(LAMBDAS), 1);
for i=1:length(LAMBDAS)
    fprintf('\tTuning PBDL: lambda = %.4f, K = %d \n', LAMBDAS(i), N_HPLANE(i));
    
    algorithm = @(y,X) PBDL(y, X, m, LAMBDAS(i), N_HPLANE(i));
    perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
    scores(i) = cross_validate(y, X, algorithm, perf_metr, n_folds);
end

[~,i] = max(scores);
lambda = LAMBDAS(i);
n_hplane = N_HPLANE(i);
% lambda = 1e-3;

fprintf('Optimal lambda : %.4f, K: %d \n', lambda, n_hplane);

[bregman_div, params] = PBDL(y, X, m, lambda, n_hplane);
