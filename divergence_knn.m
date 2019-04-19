function preds = divergence_knn(y, divs, knn_size)

n_t = size(divs,1);
max_y = max(y);

[~, inds] = sort(divs,2);

counts = zeros(n_t,max_y);

for i=1:max_y
    counts(:,i) = sum(y(inds(:,1:knn_size))==i,2);
end

[~,preds] = max(counts,[],2);