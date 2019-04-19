function out = NBR(y, X_train, S1, S2, L)

[n_train, dim] = size(X_train);

%% algorithm parameters    
m = length(S1); % number of supervision                                                                                                                       % trade-off of bias/variance     
% K = min(ceil(m^(dim/(dim+2))),100); %number of hyper-planes
K = 30;
S_pruned = union(S1,S2);

[~,S1] = ismember(S1,S_pruned);
[~,S2] = ismember(S2,S_pruned);

%% removing the unused training data and partitioning 
X = X_train(S_pruned,:); % if data already exists
n_S = length(S_pruned);
if K < n_S
    [~, c_ind] = farthest_point_clustering(X,K);                            % finding the partions
else
    K = n_S; c_ind = (1:n_S)';
end

%% preparing the optimization constants
% min 1/m sum(epsilon) + lambda L
% st: D(k,l) - D(i,j) > 1 - epsilon_ijk
% st: D(i,j) >= 0
% st: epsilon_ijkl >= 0
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
options = optimoptions('lsqlin','Algorithm','interior-point' ,'Display','off');
[z, resnorm] = lsqlin(A2,y,A,b,[],[],[],[],[],options);
params.phi = z(1:K);
params.grad = reshape(z(K+1:K*(dim+1)),[dim,K])';

out = {params,resnorm/m};