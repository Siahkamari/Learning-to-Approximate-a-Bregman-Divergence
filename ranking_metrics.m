function [auc, ave_p ] = ranking_metrics(y, divs)

divs = divs;
max_y = max(y);
n = size(y,1);

y_count = zeros(max_y,1);
for i=1:max_y
    y_count(i) = sum(y==i);
end
[~, inds] = sort(divs,2);
inds = inds(:,2:end);
y_count = y_count -1;


pre_k = cumsum(y(inds)==y,2);
pre_k = pre_k ./(1:n-1);
ave_p = mean(sum(pre_k .* y(inds)==y, 2)./y_count(y));
pre_k = mean(pre_k,1);


%% AUC
auc = zeros(n,1);
for i=1:n
    swapped_pairs = dot(y(inds(i,1:y_count(y(i)))) ~=y(i) ,...
        (y_count(y(i)) -  cumsum(y(inds(i,1:y_count(y(i)))) ==y(i))));
    auc(i) = 1 - swapped_pairs/(y_count(y(i))*(n -1 - y_count(y(i))));
end
auc = mean(auc);


