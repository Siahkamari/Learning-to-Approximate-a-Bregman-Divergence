function bregman_div = auto_tune_PBDL(y, X, m, task)


n_folds = 3;

lambdas = 10.^[-5:5];

for i=1:length(lambdas)
     fprintf('Tuning PBDL: lambda = %.5f \n', lambdas(i));
     algorithm = @(y,X) PBDL(y, X, m, lambdas(i));
     perf_metr = @(y1, X1, y2, X2, bd) performance_metric(y1, X1, y2, X2, bd, task);
     scores(i) = cross_validate(y, X, algorithm, perf_metr, n_folds);
end 

[~,i] = max(scores);
lambda = lambdas(i);

fprintf('lambda : %.5f \n', lambda);
bregman_div = PBDL(y, X, m, lambda);
