function [bregman_div, params] = auto_tune_NCA(y, X, task)
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params);
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy.
%
% Returns: a learned Bregaman divergence

n = length(y);
lambdas = linspace(0,20,5)/n;
n_folds = 3;

sigmas = 2.^(0:5);
scores = zeros(length(lambdas), 2*length(sigmas));

for j = 1:2*length(sigmas)
    
    if j<= length(sigmas)
        sigma = sigmas(j);
        kernel = "rbf";
    else 
        sigma = sigmas(j-length(sigmas));
        kernel = "poly";
    end
    
    for i=1:length(lambdas)
        fprintf('Tuning NCA: lambda = %.3f, kernel = %s, bandwidth = %.0d \n', lambdas(i), kernel, sigma);
        algorithm = @(y,X) NCA(y, X, lambdas(i), sigma, kernel);
        perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
        scores(i,j) = cross_validate(y, X, algorithm, perf_metr, n_folds);
    end 
end

[~,I] = max(scores(:));
[i, j] = ind2sub(size(scores),I);
lambda = lambdas(i);

fprintf('Optimal lambda value: %.3f\n', lambda);
if j <= length(sigmas)
    sigma = sigmas(j); kernel = "rbf";
    fprintf('Optimal kernel is RBF with sigma = %.3f\n', sigma);
else
    sigma = sigmas(j-length(sigmas)); kernel = "poly";
    fprintf('Optimal kernel is (%.3f + x^2)^2\n', sigma);
end

[bregman_div, params] = NCA(y, X, lambda, sigma , kernel);
