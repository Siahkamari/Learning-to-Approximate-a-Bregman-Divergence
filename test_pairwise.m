% Main file for learning Bregman divergences from pairwise supervision
% Output: prints and saves clustering, ranking and knn results

%% experiment parameters
clear, clc, rng(0)

% method ="Euclidean";
% method = "ITML";
method = "PBDL";    n_sup = 1000;             % number of pairwise supervisions
% method = "NCA";         % uncomment the method you desire
% method = "LMNN";

dset = 1;                 % chooses data set. 1 2 3 4 5 6 can be chosen.
task = 1;                 % 1: Bregman Clustetring:  Rand-Index
                          % 2: Bregman Clustetring:  Purity
                          % 3: Knn: Accuracy
                          % 4: Ranking: Area Under the curve
                          % 5: Ranking: Average precision
                          
train_test_ratio = 2;     % train/test data split
n_runs = 1;               % number of runs for averging

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
end

X = data.X;
y = data.y;
clear data

% making sure output labels start from 1
if min(y) == 0
    y = y + 1;
end

[n, d] = size(X);
n_train = ceil(train_test_ratio/(train_test_ratio+1)*n);

%% running
scores = zeros(n_runs,1);

for run=1:n_runs
    I_train = randsample(1:n, n_train);
    I_test = setdiff(1:n, I_train);
    y_train = y(I_train);
    X_train = X(I_train,:);
    y_test = y(I_test);
    X_test = X(I_test,:);
    
    fprintf("run %d/%d\n",run, n_runs)
    if method == "Euclidean"
        bregman_div = @(X1,X2)mahalanobis(X1, X2, eye(d), "all");
    elseif method == "ITML"
        bregman_div = ITML(y_train, X_train, task);
    elseif method == "PBDL"
        bregman_div = auto_tune_PBDL(y_train, X_train, n_sup, task);
    elseif method == "NCA"
        bregman_div =  auto_tune_NCA(y_train, X_train, task);
    elseif method == "LMNN"
        bregman_div =  auto_tune_LMNN(y_train, X_train, task);
    end
    
    scores(run) = performance_metric(y_train, X_train, y_test, X_test, bregman_div, task);
end

%% Printing
switch task
    case 1
        str = "Rand Index";
    case 2
        str = "Purity";
    case 3
        str = "K-NN Accuracy";
    case 4
        str = "Area under the curve";
    case 5
        str = "Average Precision";
end
zn = 1.96;      % 95 percent interval
fprintf("\n\n %s = %.1f  -/+  %.1f \n",str,...
    round(100*mean(scores),1),round(zn*100*std(scores)/sqrt(n_runs),1));