function divs = max_affine_bregman(X1, X2, params, mode)

%% predicting divergences
phi_X1 = max(params.phi' + X1*params.grad', [], 2);
[phi_X2, index] = max(params.phi' + X2*params.grad', [], 2);
grad_X2 = params.grad(index,:);
if nargin>3 && mode == "all"
    divs = phi_X1-phi_X2' - X1*grad_X2' + dot(X2, grad_X2, 2)';
else
    divs = phi_X1-phi_X2 - dot(grad_X2, X1-X2, 2);
end

