function dis = mahalanobis_notall(X1, X2, A)
dis = 0;
L = sqrtm(A);
X1 = X1*L;
X2 = X2*L;

for d=1:size(X1,2)
    dis = dis + (X1(:,d)-X2(:,d)).^2;
end