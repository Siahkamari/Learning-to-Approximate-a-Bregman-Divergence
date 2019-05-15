function [bregman_div, params] = auto_tune_LMNN(y, X) 
% function A = MetricLearningAutotuneKnn(metric_learn_alg, y, X, params); 
%
% Runs LMNN over various parameters of
% lambda, choosing that with the highest accuracy. 
%
% Returns: a learned Bregaman divergence



n = length(y);
xTr = X(1:floor(2*n/3),:)';
yTr = y(1:floor(2*n/3))';
xVa = X(floor(2*n/3)+1:end,:)';
yVa = y(floor(2*n/3)+1:end)';

%% tune parameters
% [K, ~, ~, maxiter] = findLMNNparams(xTr, yTr, xVa, yVa);
K = 10;
maxiter = 100;
[L, Details] = lmnnCG(X', y' ,K,'maxiter',maxiter);

A = L'*L;
bregman_div =  @(X1,X2)mahalanobis(X1,X2,A,"all");
params = A;
         