function bregman_div = euclidean_bregman(dim)
M = eye(dim);
bregman_div =  @(X1,X2)mahalanobis(X1,X2,M);