function [bregman_div, params] = auto_tune_LMNN(y, X, task) 
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% Runs LMNN over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned Bregaman divergence

lambda = 10;
n_folds = 3;

sigmas = 2.^(0:5);
scores = zeros(2*length(sigmas),1);

for i = 1:2*length(sigmas)
    
    if i<= length(sigmas)
        sigma = sigmas(i);
        kernel = "rbf";
    else 
        sigma = sigmas(i-length(sigmas));
        kernel = "poly";
    end
    fprintf('Tuning LMNN: kernel = %s, bandwidth = %.0d \n', kernel, sigma);
    algorithm = @(y,X) LMNN(y, X, lambda, sigma , kernel);
    perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
    scores(i) = cross_validate(y, X, algorithm, perf_metr, n_folds);
end

[~,i] = max(scores);

if i <= length(sigmas)
    sigma = sigmas(i); kernel = "rbf";
    fprintf('Optimal kernel is RBF with sigma = %.3f\n', sigma);
else
    sigma = sigmas(i-length(sigmas)); kernel = "poly";
    fprintf('Optimal kernel is (%.3f + x^2)^2\n', sigma);
end

[bregman_div, params] = LMNN(y, X, lambda, sigma , kernel);

         