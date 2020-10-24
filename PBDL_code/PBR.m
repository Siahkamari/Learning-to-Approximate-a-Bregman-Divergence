function out = PBR(y, X_train, S1, S2, L, trial_n)
% this is the core code for learning a bregman divergence from similarity
% values

% Input: X_train = training data. S1 and S2 are indices of random chosen
% training data from X_train. Similarity of these pairs are stored in y
% L = the hyperparameter of the learning
% trial_n = Howmany times the LP has failed

% Output: params.phi = biases of the learned convex function
% params.grad = slopes of the learned convex function

[~, dim] = size(X_train);

%% algorithm parameters
max_trials = 10;
m = length(S1);                                     % number of supervision                                     
K = 100;                                            % trade-off of bias/variance
S_pruned = union(S1,S2);

[~,S1] = ismember(S1,S_pruned);
[~,S2] = ismember(S2,S_pruned);

%% removing the unused training data and partitioning
X = X_train(S_pruned,:); % if data already exists
n_S = length(S_pruned);
if K < n_S
    [~, c_ind] = farthest_point_clustering(X,K);    % finding the partitions
else
    K = n_S; c_ind = (1:n_S)';
end

%% preparing the optimization constants
% min 1/m sum(D(i,j) - y)
% st: D(i,j) >= 0
% st:  sum (|g_i^1|, ... , |g_i^d|) < L
% z = [y, g, t, L, epsilon], y in R^n, g in R^(nd),
% t in R^nd,  L in R, epsilon in R^m

I_n_p = eye(K);

% convexity inequalities
A1 = zeros([n_S*(K-1), K*(2*dim+1)]);
count = 0;
for i=1:n_S
    for j=setdiff(1:K, c_ind(i))
        % condition that D(i,j)>0
        a = I_n_p(c_ind(i),:) - I_n_p(j,:);
        b = zeros([1,K*dim]);
        b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
        b((j-1)*dim+1:j*dim) = -X(i,:);
        count = count + 1;
        A1(count,:) = -[a,b,zeros([1,K*dim])];
    end
end
b1 = zeros(n_S*(K-1),1);

% supervision inequalities
A2 = zeros([m, K*(2*dim+1)]);
for count=1:m
    i = S1(count); j=S2(count);
    % D(i,j)
    a = I_n_p(c_ind(i),:) - I_n_p(c_ind(j),:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((c_ind(j)-1)*dim+1:c_ind(j)*dim) = -X(i,:);
    
    A2(count,:) = [a,b,zeros([1,K*dim])];
end

% condition -t_ij < g_ij < t_ij
A3 = [zeros(K*dim,K), -eye(K*dim),-eye(K*dim)];
A4 = [zeros(K*dim,K), eye(K*dim),-eye(K*dim)];
b3 = zeros(K*dim,1);
b4 = zeros(K*dim,1);

% condition  sum_{j=1}^dim t_ij < L
A5 = [zeros(K,K), zeros(K,2*K*dim)];
for i=1:K
    A5(i, K*(dim+1)+(i-1)*dim +1:K*(dim+1)+i*dim) = 1;
end
b5 = L*ones(K,1);

%% solving the LP
A = [A1;A3;A4;A5];
b = [b1;b3;b4;b5];
try
    try
        options = optimoptions('quadprog', 'Display','off');
        [z, resnorm] = quadprog_gurobi(A2'*A2,-2*y'*A2,A,b,[],[],[],[],options);
    catch
        warning('Gurobi is not installed/working, trying Matlab solvers instead');
        options = optimoptions('lsqlin', 'Algorithm', 'interior-point' ,'Display','off');
        [z, resnorm] = lsqlin(A2,y,A,b,[],[],[],[],[],options);
    end
    params.phi = z(1:K);
    params.grad = reshape(z(K+1:K*(dim+1)),[dim,K])';
    out = {params,resnorm/m};
catch
    if ~exist('trial_n', 'var')
        trial_n = 1;
    end
    if trial_n < max_trials
        warning("The random partitioning was not suitable for fitting a convex function, trying again")
        out = PBR(y(2:end), X, S1(2:end), S2(2:end), L, trial_n+1);
    else
        error("None of the random paritionings worked. Try increasing max_trials in PBR.m")
    end
end

end
