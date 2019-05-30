function [center, center_assign] = farthest_point_clustering(X,n_cluster)
% Input: X = nxd data matrix. n_cluster = number of clusters for the farthest point partitioning
% algorithm. 

% Output: center = cluster centers, center_assign = cluster assignments

[n, dim] = size(X);
center = zeros(n_cluster+1,1);
center(1) = randi(n);

for k=1:n_cluster
    dist = 0;
    for d = 1:dim
        dist = dist + (repmat(X(:,d),1,k) - repmat(X(center(1:k),d)',n,1)).^2;
    end
    [~,center(k+1)] = max(min(dist,[],2));
end
[~, center_assign] = min(dist,[],2);
center = center(1:end-1);

%% Ploting
% figure();
% scatter(X(:,1),X(:,2),40,center_assign, 'filled'); hold on
% scatter(X(center,1), X(center,2), 40, 'x')

end