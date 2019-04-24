function out = cross_validate(y, X, tCL, n_folds, knn_size)

if min(y) == 0
    y = y + 1;
end
n_cluster = int8(max(y));

[n, dim] = size(X);
if (n ~= length(y))
    disp('ERROR: num rows of X must equal length of y');
    return;
end

%permute the rows of X and y
rp = randperm(n);
y = y(rp);
X = X(rp, :);

total_purity = zeros(1, n_folds);
rand_index = zeros(1, n_folds);
acc = zeros(1,n_folds);
auc = zeros(1,n_folds);
ave_p = zeros(1,n_folds);

for i=1:n_folds
    
    %% splitting the data to test and train
    test_start = ceil(n/n_folds * (i-1)) + 1;
    test_end = ceil(n/n_folds * i);
    
    y_train = [];
    X_train = [];
    if i > 1
        y_train = y(1:test_start-1);
        X_train = X(1:test_start-1,:);
    end
    if i < n_folds
        y_train = [y_train; y(test_end+1:length(y))];
        X_train = [X_train; X(test_end+1:length(y), :)];
    end
    
    X_test = X(test_start:test_end, :);
    y_test = y(test_start:test_end);
    
    %% learning the divergence and clustering with it
    bregman_div = feval(tCL, y_train, X_train);
    [rand_index(i), total_purity(i)] = median_clustering(y_test, X_test, n_cluster, bregman_div);
    
    pred = divergence_knn(y_train, X_train, X_test, bregman_div, knn_size);
    acc(i) = mean(pred==y_test);
    
    div_rank = bregman_div(X_test, X_test);
    [auc(i), ave_p(i)] = ranking_metrics(y_test, div_rank);
    
    %% plotting the clustering assingment and phi
%     %     [~, X_pca, ~] = pca(X_test,'NumComponents',2);
%     
%     
%     y_hat = bregman_clustering(X_test, n_cluster, bregman_div);
%     X_pca = X_train;% + 0.1*randn(size(X_test));
%     figure(); hold on
%     for c=1:3
%         ind = find(y_train==c);
%         scatter(X_pca(ind,1),X_pca(ind,2),'filled');
%     end
%     
% %     % plot the mean
% %     col = get(gca,'colororder');
% %     X_bar = zeros(n_cluster,dim);                                       
% %     for c=1:3
% %         X_bar(c,:) = mean(X_test(y_hat==c,:)); 
% %         scatter(X_bar(c,1),X_bar(c,2),[], col(c,:),'filled','^');
% %     end
%     
%     
%     hold off;
%     drawnow
%     
%     plot_phi(y_train, X_train, params); drawnow

    
end
mean_total_RI = mean(rand_index);
out{1} = mean_total_RI;
out{2} = total_purity;
out{3} = rand_index;
out{4} = acc;
out{5} = auc;
out{6} = ave_p;