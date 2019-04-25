function test_pairwise(dset)

%% experiment parameters
% method ="Euclidean";
% method = "ITML";
method = "NBDL";
knn_size = 4;
n_folds = 2;
n_runs = 20;
n_sup = 14000;

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
        data = load('data/soybean-large.mat'); % 307 x 35
    case 6
        data = load('data/wdbc.mat');  % 569 x 30
    case 7
        data = load('data/breast-cancer.mat'); % 286 x 9
    case 8
        data = load('data/transfusion.mat'); % 748 x 4
    case 9
        data = load('data/abalone.mat'); % 4177 x 8
    case 10
        data = load('data/letterrecognition.mat'); % 20000 x 16
    case 11
        data = load('data/synthetic2.mat'); % 600 x 2
end

X = data.X;
y = data.y;
clear data

% X = X(1:500,:);
% y = y(1:500,:);

perm = randperm(length(y));
X = X(perm,:);
y = y(perm,:);

% [~,X,~] = pca(X, 'NumComponents', 10);

%% running
rng(0)
total_purity = zeros(n_runs,n_folds);
rand_index = zeros(n_runs,n_folds);
acc = zeros(n_runs,n_folds);
auc = zeros(n_runs,n_folds);
ave_p = zeros(n_runs,n_folds);

for run=1:n_runs
    disp(run)
    if method =="Euclidean"
        out = cross_validate(y, X, ...
            @(y,X) euclidean_bregman(size(X,2)), n_folds, knn_size);
    elseif method == "ITML"
        out = cross_validate(y, X, ...
            @(y,X) MetricLearningAutotune(@ItmlAlg, y, X), n_folds, knn_size);
    elseif method == "NBDL"
        out = cross_validate(y, X, ...
            @(y,X) auto_tune_NBDL(y, X, n_sup), n_folds, knn_size);
    end
    
    total_purity(run,:) = out{2};
    rand_index(run,:) = out{3};
    acc(run,:) = out{4};
    auc(run,:) = out{5};
    ave_p(run,:) = out{6};
end

% save(method+'dset'+num2str(dset)+'m'+num2str(n_sup)+'n_runs'+num2str(n_runs)+'.mat',...
%     'total_purity', 'rand_index','acc', 'auc', 'ave_p', 'n_folds', 'n_runs');

%% Printing
fprintf('\n\n rand index = %.1f with std %.1f \n',...
    round(100*mean(rand_index(:)),1),round(100*std(rand_index(:),1)/sqrt(n_folds*n_runs),1));

fprintf('\n\n purity = %.1f with std %.1f \n',...
    round(100*mean(total_purity(:)),1),round(100*std(total_purity(:),1)/sqrt(n_folds*n_runs),1));

fprintf('\n\n K-nn acc = %.1f with var %.1f \n',...
    round(100*mean(acc(:)),1), round(100*std(acc(:))/sqrt(n_folds*n_runs),1));

fprintf('\n\n AUC = %.1f with var %.1f \n',...
    round(100*mean(auc(:)),1), round(100*std(auc(:))/sqrt(n_folds*n_runs),1));

fprintf('\n\n ave_p = %.1f with var %.1f \n',...
    round(100*mean(ave_p(:)),1), round(100*std(ave_p(:))/sqrt(n_folds*n_runs),1));
