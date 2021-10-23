function dis = mahalanobis(X1, X2, A, mode)

% Input: X1 = n1 x d data matix. X2 = n2 x d data matix.
% A = d x d matrix of a mahalanobis distance
% if mode is all it computes the mahalanobis between all data pairs
% if mode is not all if computes the mahalanobis between each to
% corrosponding entrees in X1 and X2 (n1 should be equal n2)

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

