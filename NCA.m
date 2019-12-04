function [bregman_div, params] = NCA(y, X, lambda, sigma ,kernel)

[X_p, U] = kernel_trick_train(X, sigma, kernel);

nca = fscnca(X_p,y,'FitMethod','exact', ...
    'Solver','sgd','Lambda',lambda, ...
    'IterationLimit',30,'GradientTolerance',1e-4, ...
    'Standardize',true);
params = diag(nca.FeatureWeights);

bregman_div =  @(X1,X2)mahalanobis(kernel_trick_test(X, X1, sigma, kernel, U),...
    kernel_trick_test(X, X2, sigma, kernel, U),params,"all");

end