function dis = mahalanobis(X1, X2, A, mode)
% dis = 0;
% L = sqrtm(A);
% X1 = X1*L;
% X2 = X2*L;
% if nargin >3 && mode =="all"
%     for d=1:size(X1,2)
%         dis = dis + (X1(:,d)-X2(:,d)').^2;
%     end
% else
%     for d=1:size(X1,2)
%         dis = dis + (X1(:,d)-X2(:,d)).^2;
%     end
% end

n1 = size(X1,1);
n2 = size(X2,1);

if nargin >3 && mode =="all"
    dis = zeros(n1,n2);
    for i=1:n2
        dis(:,i) = sum(((X1-X2(i,:))*A) .*(X1-X2(i,:)),2);
    end
else
    dis = sum(((X1-X2)*A) .*(X1-X2),2);
end

