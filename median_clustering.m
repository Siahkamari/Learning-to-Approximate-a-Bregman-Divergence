function [median_rand_index, median_total_purity] = median_clustering(y, X, n_cluster, bregman_div)

num_run = 20;
total_purity = zeros(num_run,1);
rand_index = zeros(num_run,1);

for run = 1:num_run
    y_hat = bregman_clustering(X, n_cluster, bregman_div);
    
    %% computing purity
    n_test = length(y);
    map = zeros(n_cluster,1);
    purity = zeros(n_cluster,1);
    count = zeros(n_cluster,1);
    for c=1:n_cluster
        ind = find(y_hat==c);
        map(c) = mode(y(ind));
        purity(c) = mean(y(ind) == map(c));
        count(c) = length(ind);
    end
    total_purity(run) = nansum(purity.*count)/n_test;
    
    %% computing rand index
    for i=1:n_test
        rand_index(run) = rand_index(run)+...
            sum((y_hat(i)==y_hat(setdiff(1:n_test, i)))==...
            (y(i)==y(setdiff(1:n_test, i))));
    end
    rand_index(run) = rand_index(run)/(n_test*(n_test-1));
end

median_rand_index = median(rand_index);
median_total_purity = median(total_purity);