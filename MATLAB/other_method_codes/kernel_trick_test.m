function X_p = kernel_trick_test(XTr, XTst, sigma, kernel, U)

[nTr, ~] = size(XTr);
[nTst, ~] = size(XTst);

deltaXTst = zeros(nTst, nTr);
for i=1:nTst
   deltaXTst(i,:) = sum((XTst(i,:)-XTr).^2, 2)';
end

if kernel == "rbf"
    KTst = exp(-deltaXTst/(2*sigma^2));
elseif kernel == "poly"
    KTst = (sigma+deltaXTst).^2;
end

K_tild = KTst - 1/nTst*ones(nTst)*KTst - KTst*1/nTr*ones(nTr) ...
    + 1/nTst*ones(nTst)*KTst*1/nTr*ones(nTr);

X_p = K_tild*U;

end