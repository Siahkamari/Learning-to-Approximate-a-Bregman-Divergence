function params = PBDLK_core(X_train, S1, S2, S3, S4, lambda, K, trial_n)
% this is the core code for learning a bregman divergence from pairwise
% inequalities.

% Input: X_train = training data. S1 and S2 are indices of random chosen
% training data from the same cluster. S3 and S4 are form different
% clusters. 
% lambda = the hyperparameter of the learning
% K = number of hyperplanes to use
% trial_n = Howmany times the LP has failed
max_trials = 10; 

% Output: params.phi = biases of the learned convex function
% params.grad = slopes of the learned convex function


%% algorithm parameters  
[~, dim] = size(X_train);       
m = length(S1);
S_pruned = union(union(union(S1,S2),S3),S4);

[~,S1] = ismember(S1, S_pruned);
[~,S2] = ismember(S2, S_pruned);
[~,S3] = ismember(S3, S_pruned);
[~,S4] = ismember(S4, S_pruned);

%% removing the unused training data and partitioning
X = X_train(S_pruned,:); % if data already exists
n_S = length(S_pruned);
if K < n_S
    [~, c_ind] = farthest_point_clustering(X,K);                            % finding the partitions
else
    K = n_S; c_ind = (1:n_S)';
end

%% preparing the optimization constants
% min 1/m sum(epsilon) + lambda L
% st: D(k,l) - D(i,j) > 1 - epsilon_ijk
% st: D(i,j) >= 0
% st: epsilon_ijkl >= 0
% st:  sum (|a_i^1|, ... , |a_i^d|) < L
% z = [b, a, t, L, epsilon], b in R^n, a in R^(nd),
% t in R^nd,  L in R, epsilon in R^m

I_n_p = eye(K);

% convexity inequalities
A1 = zeros([n_S*(K-1), K*(2*dim+1)+1+m]);
count = 0;
for i=1:n_S
    for j=setdiff(1:K, c_ind(i))
        % condition that D(i,j)>0
        a = I_n_p(c_ind(i),:) - I_n_p(j,:);
        b = zeros([1, K*dim]);
        b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
        b((j-1)*dim+1:j*dim) = -X(i,:);
        count = count + 1;
        A1(count,:) = -[a,b,zeros([1,K*dim]),0,zeros(1,m)];
    end
end
b1 = zeros(n_S*(K-1),1);

% supervision inequalities
A2 = zeros([m, K*(2*dim+1)+1+m]);
for count=1:m
    i = S1(count); j=S2(count); k=S3(count); l=S4(count);
    % D(i,j)
    a = I_n_p(c_ind(i),:) - I_n_p(c_ind(j),:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((c_ind(j)-1)*dim+1:c_ind(j)*dim) = -X(i,:);
    a1 = [a,b,zeros([1,K*dim]),0,zeros(1,m)];
    
    % D(k,l)
    a = I_n_p(c_ind(k),:) - I_n_p(c_ind(l),:);
    b = zeros([1,K*dim]);
    b((c_ind(k)-1)*dim+1:c_ind(k)*dim) = X(k,:);
    b((c_ind(l)-1)*dim+1:c_ind(l)*dim) = -X(k,:);
    a2 = [a,b,zeros([1,K*dim]),0,zeros(1,m)];
    
    % condition that D(k,l) - D(i,j) > 1-eps_ijkl
    A2(count,:) = a1 - a2;
    A2(count,K*(2*dim+1)+1+count) = -1;
end
b2 = -1*ones(m,1);

% condition -t_ij < a_ij < t_ij
A3 = [zeros(K*dim,K), -eye(K*dim),-eye(K*dim),zeros(K*dim,1),zeros(K*dim,m)];
A4 = [zeros(K*dim,K), eye(K*dim),-eye(K*dim),zeros(K*dim,1),zeros(K*dim,m)];
b3 = zeros(K*dim,1);
b4 = zeros(K*dim,1);

% condition  sum_{j=1}^dim t_ij < L
A5 = [zeros(K,K), zeros(K,2*K*dim), -ones(K,1), zeros(K,m)];
for i=1:K
    A5(i, K*(dim+1)+(i-1)*dim +1:K*(dim+1)+i*dim) = 1;
end
b5 = zeros(K,1);

% condition that eps_i  > 0
A6 = [zeros(m,K*(2*dim+1)+1), -eye(m)];
b6 = zeros(m,1);

% price vectors
C1 = [zeros(1,K*(2*dim+1)),0,ones(1,m)];                                    % sum (eps_i)
C2 = [zeros(1,K*(2*dim+1)),1,zeros(1,m)];                                   % L
C = C1/m + lambda*C2;

%% solving the LP
A = [A1;A2;A3;A4;A5;A6];
b = [b1;b2;b3;b4;b5;b6];

try
    options = optimoptions('linprog','Display','off');
    z = linprog_gurobi(C,A,b,[],[],[],[],options);
catch
    warning('Gurobi is not installed/working, trying Matlab solvers instead');
    options1 = optimoptions('linprog','Algorithm','interior-point' ,'Display','final',...
        'MaxIterations',1000);
    options2 = optimoptions('linprog','Algorithm','dual-simplex' ,'Display','final');
    options3 = optimoptions('linprog','Algorithm','interior-point-legacy' ,'Display','final');
    
    [z, ~, exitFlag] = linprog(C,A,b,[],[],[],[],options1);
    if exitFlag~=1
        [z, ~, exitFlag] = linprog(C,A,b,[],[],[],[],options2);
    end
    if exitFlag~=1
        z = linprog(C,A,b,[],[],[],[],options3);
    end
end
try
    params.phi = z(1:K);
    params.grad = reshape(z(K+1:K*(dim+1)),[dim,K])';
catch
    if ~exist('trial_n', 'var') 
        trial_n = 1;
    end
    if trial_n < max_trials
        warning("The random partitioning was not suitable for fitting a convex function, trying again") 
        params = PBDLL1(X, S1, S2, S3, S4, lambda, K, trial_n+1);
    else
        error("None of the random paritionings worked. Try increasing max_trials in PBDLL1.m") 
    end
end
end