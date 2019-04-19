close all; clear, clc

dset=3;

%% loading data

switch dset
    case 1
        data =  load('data/iris.mat');
    case 2
        data =  load('data/ionosphere.mat');
    case 3
        data =  load('data/balance-scale.mat');
    case 4
        data =  load('data/wine.mat');
    case 5
        data =  load('data/soybean-large.mat');
end

X = data.X;
y = data.y;
clear data

% [~,X,~] = pca(X, 'NumComponents',10);

%% experiment parameters
num_folds = 2;
knn_size = 4;
num_runs = 20;
% method ="Euclidean";
% method = "ITML";
method = "NBDL";

%% running
rng(0)
acc = zeros(num_runs,1);
for run = 1:num_runs
    disp(run)
    acc(run) = cross_validate_knn(y, X,...
        @(y,X) MetricLearningAutotuneKnn(@ItmlAlg, y, X), num_folds, knn_size, method);
end

fprintf('\n\n knn cross-validated error = %f with var %f \n',...
    100*mean(acc), 100*std(acc)/sqrt(num_folds*num_runs));
