function preds = divergence_knn(y_train, X_train, X_test, bregman_div, knn_size)

divs = bregman_div(X_test, X_train);

n_t = size(divs,1);
max_y = max(y_train);

[~, inds] = sort(divs,2);

counts = zeros(n_t,max_y);

for i=1:max_y
    counts(:,i) = sum(y_train(inds(:,1:knn_size))==i,2);
end

[~,preds] = max(counts,[],2);