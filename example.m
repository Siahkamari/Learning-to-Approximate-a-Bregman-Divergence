% Main file for learning Bregman divergences from pairwise supervision
% Output: prints and saves clustering, ranking and knn results
clear, clc

%% experiment parameters
rng(0)

% method ="Euclidean";
% method = "ITML";
method = "PBDL";                
% method = "NCA";         % uncomment the method you desire
% method = "GMML";  
dset = 1;                 % chooses data set. 1 2 3 4 5 6 can be chosen.
task = 5;                 % 1: Bregman Clustetring:  Rand-Index
                          % 2: Bregman Clustetring:  Purity
                          % 3: Knn: Accuracy
                          % 4: Ranking: Area Under the curve
                          % 5: Ranking: Average precision
                          
n_sup = 2000;             % number of pairwise supervisions ONLY for PBDL and GMML
train_test_ratio = 2;     % train/test data split
n_runs = 2;               % number of runs for averging

%% loading data
switch dset
    case 1
        data = load('data/iris.mat'); % 150 x 4
    case 2
        data = load('data/balance-scale.mat'); % 625 x 4
    case 3
        data = load('data/wine.mat'); % 178 x 13
    case 4
        data = load('data/transfusion.mat'); % 748 x 4
    case 5
        data = load('data/synthetic2.mat'); % 600 x 2
end

X = data.X;
y = data.y;

clear data

if min(y) == 0
    y = y+1;
end
[n, d] = size(X);

n_train = ceil(train_test_ratio/(train_test_ratio+1)*n);

%% running
scores = zeros(5,2,n_runs);

for run=1:n_runs
    rng(run)

    I_train = randsample(n,n_train);
    I_test = setdiff(1:n, I_train);
    X_train = X(I_train,:);
    y_train = y(I_train);
    X_test = X(I_test,:);
    y_test = y(I_test);
    
    fprintf("\n ---------- run %d/%d ----------- \n",run, n_runs)
    cf = cd('./utils'); addpath(genpath(pwd)); cd(cf)
    if method == "Euclidean"
        cf = cd('./other_method_codes'); addpath(genpath(pwd)); cd(cf)
        bregman_div = @(X1,X2)mahalanobis(X1, X2, eye(d), "all");
    elseif method == "ITML"
        cf = cd('./other_method_codes/ITML_code'); addpath(genpath(pwd)); cd(cf)
        bregman_div = MetricLearningAutotune(@ItmlAlg, y_train, X_train ,[], task);
    elseif method == "PBDL"
        cf = cd('./PBDL_code'); addpath(genpath(pwd)); cd(cf)
        bregman_div = auto_tune_PBDL(y_train, X_train, n_sup, task);
    elseif method == "NCA"
        cf = cd('./other_method_codes'); addpath(genpath(pwd)); cd(cf)
        bregman_div =  auto_tune_NCA(y_train, X_train, task);
    elseif method == "GMML"
        cf = cd('./other_method_codes'); addpath(genpath(pwd)); cd(cf)
        bregman_div = auto_tune_GMML(y_train, X_train,  n_sup, task);
    end
    
    scores(:, :, run) = performance_metric_all(y_train, X_train, y_test, X_test, bregman_div);
end

str = "dset = " + num2str(dset)+ ", method = "+ method + ", n_runs = " +...
    num2str(n_runs)+ ", nsup = " + num2str(n_sup);

save("scores_"+str+".mat",'scores')
fprintf("\n"+str+"\n")

mean_scores = mean(scores,3);
std_scores = std(scores,0,3);

%% Printing
fprintf("\n\n -.-.-.-.-.-.  Train Performance  .-.-.-.-.-.-.- \n")
for task = 1:5
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
    fprintf("\n\n %s = %.1f +- %.1f \n",str,round(100*mean_scores(task,1),1),round(100*std_scores(task,1)/sqrt(n_runs),1))
end
fprintf("\n")

%% Printing
fprintf("\n -.-.-.-.-.-.  Test Performance .-.-.-.-.-.-.- \n")
for task = 1:5
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
    fprintf("\n\n %s = %.1f +- %.1f \n",str,round(100*mean_scores(task,2),1),round(100*std_scores(task,2)/sqrt(n_runs),1))
end
