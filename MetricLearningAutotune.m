function bregman_div = MetricLearningAutotune(metric_learn_alg, y, X, params)
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

out = cell(length(gammas), 1);
accs = zeros(length(gammas), 1);
for i=1:length(gammas)
%     fprintf('\tTuning burg kernel learning: gamma = %f', gammas(i));
    params.gamma = gammas(i); 
    out{i} = cross_validate(y, X, @(y,X) MetricLearning(metric_learn_alg, y, X, A0, params), 2, params.k);
    accs(i) = out{i}{1};
end

[~,i] = max(accs);
gamma = gammas(i);
fprintf('Optimal gamma value: %f\n', gamma);
params.gamma = gamma;
bregman_div  = MetricLearning(metric_learn_alg, y, X, A0, params);