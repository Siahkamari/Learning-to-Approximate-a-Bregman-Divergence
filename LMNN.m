function [bregman_div, params] = LMNN(y, X, K, sigma , kernel)

cf = cd('./LMNN_code'); setpaths3(); cd(cf)

[X_p, U] = kernel_trick_train(X, sigma, kernel);

maxiter = 100;
[L, ~] = lmnnCG(X_p', y' ,K,'maxiter',maxiter);
A = L'*L;
params = A;

bregman_div =  @(X1,X2)mahalanobis(kernel_trick_test(X, X1, sigma, kernel, U),...
    kernel_trick_test(X, X2, sigma, kernel, U),params,"all");
