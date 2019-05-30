function [auc, ave_p ] = ranking_metrics(y, divs)
% Input: y are the class labels of training data a n x 1 vector
% divs is a n x n matrix where divs(i,j) is divergence of X_i from X_j

% Output: rankings mesures, auc = area under the curve.
% ave_p = Average precision

max_y = max(y);
n = size(y,1);

y_count = zeros(max_y,1);
for i=1:max_y
    y_count(i) = sum(y==i);
end
[~, inds] = sort(divs,2);
inds = inds(:,2:end);
y_count = y_count - 1;


%% average precision
pre_k = cumsum(y(inds)==y,2);
pre_k = pre_k ./(1:n-1);
ave_p = nanmean(sum(pre_k .* (y(inds)==y), 2)./y_count(y));
% pre_k = mean(pre_k,1);


%% AUC
swapped_pairs = sum( (y(inds) ~=y) .*(y_count(y) -  cumsum(y(inds) ==y, 2)), 2);
auc = 1 - swapped_pairs./(y_count(y).*(n - 1 - y_count(y)));

auc = nanmean(auc);


