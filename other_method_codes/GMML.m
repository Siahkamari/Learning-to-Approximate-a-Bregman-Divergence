function [bregman_div, params] = GMML(y, X, m, lambda, t)

[S1, S2, S3, S4] = get_supervision_pairs(m, y);

Sim = 0;
Dim = 0;
for i=1:length(S1)
    Sim = Sim + (X(S1(i,:)) - X(S2(i,:)))'*(X(S1(i,:)) - X(S2(i,:)));
    Dim = Dim + (X(S3(i,:)) - X(S4(i,:)))'*(X(S3(i,:)) - X(S4(i,:)));
end

% A0 = eye(size(X,2));
% A = inv(Sim + lambda * A0);
% B = Dim + lambda * A0;
% 
% 
% [V, D] = eig(A);
% A12 = V * (D.^0.5)/ V;
% invA12 = inv(A12);
% 
% Sigma = invA12*B*invA12;
% 
% [V, D] = eig(Sigma);
% Sigma2 = V * (D.^t)/ V;
% 
% params = A12 * Sigma2 * A12;



[V, D] = eig(Sim);
A12 = V * (D.^0.5)/ V;
invA12 = inv(A12);

Sigma = A12*Dim*A12;

[V, D] = eig(Sigma);
Sigma2 = V * (D.^1/2)/ V;

params = invA12 * Sigma2 * invA12;

bregman_div = @(X1,X2)mahalanobis(X1, X2, params, "all");

end
