function [bregman_div, params] = NCA(y, X, lambda)

nca = fscnca(X,y,'FitMethod','exact', ...
    'Solver','sgd','Lambda',lambda, ...
    'IterationLimit',30,'GradientTolerance',1e-4, ...
    'Standardize',true);
params = diag(nca.FeatureWeights);
bregman_div =  @(X1,X2)mahalanobis(X1,X2,params,"all");

end