function bregman_div = auto_tune_NBDL(y, X, m) 
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% Runs NBDL over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned Bregaman divergence



% define lambda the multiplier of the regularization
% lambdas = 10.^(-5:0);
% knn_size = 4;
% out = cell(length(gammas), 1);
% accs = zeros(length(gammas), 1);
% for i=1:length(lambdas)
%     fprintf('\tTuning NBDL: lambda = %f', lambdas(i));
%     out{i} = cross_validate(y, X, @(y,X) NBDL(y, X, m, lambdas(i)), 2, knn_size);
%     accs(i) = out{i}{1};
% end
% 
% [~,i] = max(accs);
% lambda = lambdas(i);
lambda = 0.001;
% fprintf('\tOptimal lambda value: %f', lambda);                        
bregman_div = NBDL(y, X, m, lambda);