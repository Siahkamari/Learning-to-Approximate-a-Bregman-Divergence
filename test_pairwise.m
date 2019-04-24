close all; clear, clc

%% experiment parameters
method ="Euclidean";
method = "ITML";
% method = "NBDL";

dset = 2;

knn_size = 4;
num_folds = 2;
num_runs = 20;
num_supervision = 1000;

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
    case 6
        data =  load('data/letterrecognition.mat');
    case 7
        data =  load('data/synthetic2.mat');
end

X = data.X;
y = data.y;
clear data

% X = X(1:500,:);
% y = y(1:500,:);

% [~,X,~] = pca(X, 'NumComponents', 10);

%% running
rng(0)
total_purity = zeros(num_runs,num_folds);
rand_index = zeros(num_runs,num_folds);
acc = zeros(num_runs,num_folds);
auc = zeros(num_runs,num_folds);
ave_p = zeros(num_runs,num_folds);

for run=1:num_runs
    disp(run)
    if method =="Euclidean"
        out = cross_validate(y, X, ...
            @(y,X) euclidean_bregman(size(X,2)), num_folds, knn_size);
    elseif method == "ITML"
        out = cross_validate(y, X, ...
            @(y,X) MetricLearningAutotune(@ItmlAlg, y, X), num_folds, knn_size);
    elseif method == "NBDL"
        out = cross_validate(y, X, ...
            @(y,X) auto_tune_NBDL(y, X, num_supervision), num_folds, knn_size);
    end
    
total_purity(run,:) = out{2};
rand_index(run,:) = out{3};
acc(run,:) = out{4};
auc(run,:) = out{5};
ave_p(run,:) = out{6};
end

%% Printing
fprintf('\n\n clustering rand index median over runs = %f with std %f \n',...
    100*mean(rand_index(:)),100*std(rand_index(:),1)/sqrt(num_folds*num_runs));

fprintf('\n\n clustering purity median over runs = %f with std %f \n',...
    100*mean(total_purity(:)),100*std(total_purity(:),1)/sqrt(num_folds*num_runs));

fprintf('\n\n K-nn  error = %f with var %f \n',...
    100*mean(acc(:)), 100*std(acc(:))/sqrt(num_folds*num_runs));

fprintf('\n\n AUC = %f with var %f \n',...
    100*mean(auc(:)), 100*std(auc(:))/sqrt(num_folds*num_runs));

fprintf('\n\n average precision = %f with var %f \n',...
    100*mean(ave_p(:)), 100*std(ave_p(:))/sqrt(num_folds*num_runs));
