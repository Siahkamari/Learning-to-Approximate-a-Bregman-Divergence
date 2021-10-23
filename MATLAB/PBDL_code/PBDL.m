function [bregman_div, params] = PBDL(y, X, m, lambda)

% [S1, S2, S3, S4] = get_supervision_pairs(m, y);
% K = 100;
% params = PBDLK_core(X, S1, S2, S3, S4, lambda, K);
params = PBDL_core_triplets(X, S1, S2, S3, S4, lambda);
% params = PBDL_core_pairs(y, X, lambda);
bregman_div = @(X1,X2)max_affine_bregman(X1,X2,params, "all");

% plot_phi(y, X, params)

end
