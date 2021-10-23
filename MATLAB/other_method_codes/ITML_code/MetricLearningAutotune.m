function [bregman_div, A] = MetricLearningAutotune(metric_learn_alg, y, X, params, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% metric_learn_alg: 
% Runs information-theoretic metric learning over various parameters of
% gamma, choosing that with the highest accuracy. 
%
% Returns: Mahalanobis matrix A for learned distance metric


if (~exist('params'))
    params = struct();
end
params = SetDefaultParams(params);

% regularize to the identity matrix
A0 = eye(size(X, 2));

% define gamma values for slack variables
gammas = 10.^(-4:4);
n_folds = 3;
scores = zeros(length(gammas), 1);
perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
for i=1:length(gammas)
%     fprintf('\tTuning burg kernel learning: gamma = %f', gammas(i));
    params.gamma = gammas(i); 
    scores(i) = cross_validate(y, X, @(y,X) MetricLearning(metric_learn_alg, y, X, A0, params), perf_metr, n_folds);
end

[~,i] = max(scores);
gamma = gammas(i);
fprintf('Optimal gamma value: %.4f\n', gamma);
params.gamma = gamma;
[bregman_div, A]  = MetricLearning(metric_learn_alg, y, X, A0, params);