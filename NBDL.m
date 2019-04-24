function bregman_div = NBDL(y, X, m, lambda)

[S1, S2, S3, S4] = get_supervision_pairs(m, y);
params = NBDLL1(X, S1, S2, S3, S4, lambda);
bregman_div = @(X1,X2)max_affine_bregman(X1,X2,params, "all");

end
