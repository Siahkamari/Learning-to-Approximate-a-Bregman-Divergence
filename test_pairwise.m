function test_pairwise(dset, task)

%% experiment parameters
% method ="Euclidean";
% method = "ITML";
method = "NBDL";
% method = "NCA";
% method = "LMNN";

knn_size = 5;
n_folds = 3;
n_runs = 50;
n_sup = 15000;
% task = 1;     % performance measure for cross_validation hyper-parameter search

%% loading data
switch dset
    case 1
        data = load('data/iris.mat'); % 150 x 4
    case 2
        data = load('data/ionosphere.mat'); % 351 x 34
    case 3
        data = load('data/balance-scale.mat'); % 625 x 4
    case 4
        data = load('data/wine.mat'); % 178 x 13
    case 5
        data = load('data/transfusion.mat'); % 748 x 4
    case 6
        data = load('data/synthetic2.mat'); % 600 x 2
    case 7
        data = load('data/wdbc.mat');  % 569 x 30
    case 8
        data = load('data/breast-cancer.mat'); % 286 x 9
    case 9
        data = load('data/abalone.mat'); % 4177 x 8
    case 10
        data = load('data/soybean-large.mat'); % 307 x 35
    case 11
        data = load('data/letterrecognition.mat'); % 20000 x 16
end

X = data.X;
y = data.y;
clear data

% perm = randperm(length(y));
% X = X(perm,:);
% y = y(perm);

% X = X(1:500,:);
% y = y(1:500,:);

% [~,X,~] = pca(X, 'NumComponents', 10);

%% running
rng(0)
total_purity = zeros(n_runs,n_folds);
rand_index = zeros(n_runs,n_folds);
acc = zeros(n_runs,n_folds);
auc = zeros(n_runs,n_folds);
ave_p = zeros(n_runs,n_folds);

for run=1:n_runs
    clc
    fprintf("run %d/%d\n",run, n_runs)
    if method == "Euclidean"
        out = cross_validate(y, X, ...
            @(y,X) euclidean_bregman(size(X,2)), n_folds, knn_size, task);
    elseif method == "ITML"
        out = cross_validate(y, X, ...
            @(y,X) MetricLearningAutotune(@ItmlAlg, y, X,[], task), n_folds, knn_size, task);
    elseif method == "NBDL"
        out = cross_validate(y, X, ...
            @(y,X) auto_tune_NBDL(y, X, n_sup, task), n_folds, knn_size, task);
    elseif method == "NCA"
        out = cross_validate(y, X, ...
            @(y,X) auto_tune_NCA(y, X, task), n_folds, knn_size, task);
    elseif method == "LMNN"
        out = cross_validate(y, X, ...
            @(y,X) auto_tune_LMNN(y, X), n_folds, knn_size, task);
    end
    
    rand_index(run,:) = out{2};
    total_purity(run,:) = out{3};
    acc(run,:) = out{4};
    ave_p(run,:) = out{5};
    auc(run,:) = out{6};
end

save("dset"+num2str(dset)+"_"+method+"_m"+num2str(n_sup)+"_n_runs"+num2str(n_runs)+...
    "_task"+num2str(task)+".mat",'rand_index','total_purity','acc', 'ave_p', 'auc', 'n_folds', 'n_runs');

%% Printing
zn = 1.96;      % 95 percent interval
fprintf("\n\n Rand Index = %.1f  -/+  %.1f \n",...
    round(100*mean(rand_index(:)),1),round(zn*100*std(rand_index(:),1)/sqrt(n_folds*n_runs),1));

fprintf("\n\n Purity = %.1f  -/+  %.1f \n",...
    round(100*mean(total_purity(:)),1),round(zn*100*std(total_purity(:),1)/sqrt(n_folds*n_runs),1));

fprintf("\n\n K-NN Accuracy = %.1f  -/+  %.1f \n",...
    round(100*mean(acc(:)),1), round(zn*100*std(acc(:))/sqrt(n_folds*n_runs),1));

fprintf("\n\n Ave_P %.1f  -/+  %.1f \n",...
    round(100*mean(ave_p(:)),1), round(zn*100*std(ave_p(:))/sqrt(n_folds*n_runs),1));

fprintf("\n\n AUC = %.1f  -/+  %.1f \n",...
    round(100*mean(auc(:)),1), round(zn*100*std(auc(:))/sqrt(n_folds*n_runs),1));
