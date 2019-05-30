function [bregman_div, params] = PBDL(y, X, m, lambda, K)

[S1, S2, S3, S4] = get_supervision_pairs(m, y);
params = PBDLL1(X, S1, S2, S3, S4, lambda, K);
bregman_div = @(X1,X2)max_affine_bregman(X1,X2,params, "all");

end
