close all; clear, clc

%% loading data
dset = 4;

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
    case 6
        data =  load('data/synthetic2.mat');
end

X = data.X;
y = data.y;
clear data

% [~,X,~] = pca(X, 'NumComponents', 10);

%% experiment parameters
num_folds = 2;
num_runs = 20;
method ="Euclidean";
method = "ITML";
method = "NBDL";

%% running
rng(0)
total_purity = zeros(num_runs,num_folds);
rand_index = zeros(num_runs,num_folds);
for run=1:num_runs
    disp(run)
    if method =="Euclidean"
        [~, total_purity(run,:), rand_index(run,:)] = cross_validate_clustering(y, X, ...
            @(y,X) euclidean_bregman(size(X,2)), num_folds);
    elseif method == "ITML"
        [~, total_purity(run,:), rand_index(run,:)] = cross_validate_clustering(y, X, ...
            @(y,X) MetricLearningAutotuneClustering(@ItmlAlg, y, X), num_folds);
    elseif method == "NBDL"
        [~, total_purity(run,:), rand_index(run,:)] = cross_validate_clustering(y, X, ...
            @(y,X) auto_tune_NBDL_clustering(y, X), num_folds);
    end
end

fprintf('\n\n cross-validated clustering rand index median over runs = %f with std %f \n',...
    mean(100*rand_index(:)),100*std(rand_index(:),1)/sqrt(num_folds*num_runs));

fprintf('\n\n cross-validated clustering purity median over runs = %f with std %f \n',...
    mean(100*total_purity(:)),100*std(total_purity(:),1)/sqrt(num_folds*num_runs));
