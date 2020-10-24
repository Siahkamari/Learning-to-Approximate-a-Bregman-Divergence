function score = performance_metric_all(y_train, X_train, y_test, X_test, bregman_div)

knn_size = 5;
n_cluster = int8(max([y_train;y_test]));


%% Train
 score(1,1)=inf;, score(2,1) = inf;
% clustering - Rand-Index, Purity
[score(1,1), score(2,1)] =...
    median_clustering(y_train, X_train, n_cluster, bregman_div);

% knn - accuracy
pred = divergence_knn(y_train, X_train, X_train, bregman_div, knn_size);
score(3,1) = mean(pred == y_train);

% ranking - Area under the curve, , Average precision
div_rank = bregman_div(X_train, X_train);
[score(4,1), score(5,1)] = ranking_metrics(y_train, div_rank);


%% Test
 score(1,2)=inf;, score(2,2) = inf;
%  % clustering - Rand-Index, Purity
[score(1,2), score(2,2)] =...
     median_clustering(y_test, X_test, n_cluster, bregman_div);

% knn - accuracy
pred = divergence_knn(y_train, X_train, X_test, bregman_div, knn_size);
score(3,2) = mean(pred == y_test);

% ranking - Area under the curve, , Average precision
div_rank = bregman_div(X_test, X_test);
[score(4,2), score(5,2)] = ranking_metrics(y_test, div_rank);