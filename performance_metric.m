function score = performance_metric(y_train, X_train, y_test, X_test, bregman_div, task)

knn_size = 5;
if min([y_train;y_test]) == 0
    y_train = y_train + 1;
    y_test = y_test + 1;
end
n_cluster = int8(max([y_train;y_test]));

switch task
    case 1 % clustering - Rand-Index
    [score, ~] =...
        median_clustering(y_test, X_test, n_cluster, bregman_div);
    case 2 % clustering - Purity
    [~, score] =...
        median_clustering(y_test, X_test, n_cluster, bregman_div);
    case 3  % knn - accuracy
    pred = divergence_knn(y_train, X_train, X_test, bregman_div, knn_size);
    score = mean(pred == y_test);
    case 4  % ranking - Area under the curve
    div_rank = bregman_div(X_test, X_test);
    [score, ~] = ranking_metrics(y_test, div_rank);
    case 5  % ranking - Average precision
    div_rank = bregman_div(X_test, X_test);
    [~, score] = ranking_metrics(y_test, div_rank);      
end

end