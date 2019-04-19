function bregman_div = auto_tune_NBDL_clustering(y, X) 
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned Bregaman divergence



% define lambda the multiplier of the regularization
lambdas = 10.^(-5:0);

m = 12000;
rand_index = zeros(length(lambdas), 1);
for i=1:length(lambdas)
    fprintf('\tTuning NBDL: lambda = %f', lambdas(i));
    rand_index(i) = cross_validate_clustering(y, X, @(y,X) NBDL(y, X, m, lambdas(i)), 2);
end

[~,i] = max(rand_index);
lambda = lambdas(i);
% lambda = 0.001;
fprintf('\tOptimal lambda value: %f', lambda);

m = 12000;                          
bregman_div = NBDL(y, X, m, lambda);