function [X_p, U] = kernel_trick_train(XTr, sigma, kernel)

[nTr, ~] = size(XTr);

deltaX = zeros(nTr,nTr);
for i=1:nTr
   deltaX(i,:) = sum((XTr(i,:)-XTr).^2, 2)';
end

if kernel == "rbf"
    K = exp(-deltaX/(2*sigma^2));
elseif kernel == "poly"
    K = (sigma+deltaX).^2;  
end

K_tild = K - 2/nTr*ones(nTr)*K + 1/nTr*ones(nTr)*K*1/nTr*ones(nTr);

[U,S,~] = svd(K_tild);

U = U ./(sqrt(diag(S)'));

U = U(:,1:floor(nTr/2));
X_p = K_tild*U;

end