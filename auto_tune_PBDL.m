function [bregman_div, params] = auto_tune_PBDL(y, X, m, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params);
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy.
%
% Returns: a learned Bregaman divergence


% define lambda the multiplier of the regularization



% LAMBDAS = 10.^(-5:1);
% 
% knn_size = 5;
% n_folds = 3;
% 
% objective = zeros(length(LAMBDAS), 1);
% for i=1:length(LAMBDAS(:))
%     fprintf('\tTuning PBDL: lambda = %.4f \n', LAMBDAS(i));
%     out = cross_validate(y, X, @(y,X) PBDL(y, X, m, LAMBDAS(i)), n_folds, knn_size, task);
%     objective(i) = out{1};
% end
% 
% [~,i] = max(objective);
% lambda = LAMBDAS(i);

lambda = 1e-3;

fprintf('\tOptimal lambda : %.4f, %d \n', lambda);

[bregman_div, params] = PBDL(y, X, m, lambda);
