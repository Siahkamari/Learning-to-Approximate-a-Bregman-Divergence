function divs = max_affine_bregman(X1, X2, params, mode)
% Input: X1 = n1 x d data matix. X2 = n2 x d data matix.
% params = slopes and biases of a max affine function
% if mode is all it computes the Bregman divergence between all data pairs
% if mode is not all, it computes the Bregman divergence between each two
% corrosponding entrees in X1 and X2 (thefore n1 should be equal to n2)

%% predicting divergences
if iscell(params)
    param = params{1};
    n_max_affine = length(params);
else
    param = params;
end

phi_X1 = max(param.phi' + X1*param.grad', [], 2);
[phi_X2, index] = max(param.phi' + X2*param.grad', [], 2);
grad_X2 = param.grad(index,:);

if iscell(params)
    for i=2:n_max_affine
        param = params{i};
        phi_X1 = phi_X1 + max(param.phi' + X1*param.grad', [], 2);
        [phi_X2, index] = max(param.phi' + X2*param.grad', [], 2);
        grad_X2 = grad_X2 + param.grad(index,:);
    end
    phi_X1 = phi_X1/n_max_affine;
    grad_X2 = grad_X2/n_max_affine;
end

if nargin>3 && mode == "all"
    divs = phi_X1-phi_X2' - X1*grad_X2' + dot(X2, grad_X2, 2)';
else
    divs = phi_X1-phi_X2 - dot(grad_X2, X1-X2, 2);
end

