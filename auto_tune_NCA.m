function [bregman_div, params] = auto_tune_NCA(y, X, task) 
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned Bregaman divergence



n = length(y);
lambdas = linspace(0,20,20)/n;
knn_size = 5;
n_folds = 3;

out = cell(length(lambdas), 1);
accs = zeros(length(lambdas), 1);
for i=1:length(lambdas)
%     fprintf('Tuning NCA: lambda = %f \n', lambdas(i));
    out{i} = cross_validate(y, X, @(y,X) NCA(y, X, lambdas(i)), n_folds, knn_size, task);
    accs(i) = out{i}{1};
end

[~,i] = max(accs);
lambda = lambdas(i);

fprintf('Optimal lambda value: %f\n', lambda); 

[bregman_div, params] = NCA(y, X, lambda);
         