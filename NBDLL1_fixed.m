function params = NBDLL1(X_train, S1, S2, S3, S4, lambda)

[n_train, dim] = size(X_train);

%% algorithm parameters                                                     % number of supervision                                                             % noise of supervision                                                                                                                   % trade-off of bias/variance                                                           % number of hyper-planes
% K = min(ceil(n_train^(dim/(dim+2))),100);
K = 100;
m = length(S1);
L = 100;

S_pruned = union(union(union(S1,S2),S3),S4);

[~,S1] = ismember(S1, S_pruned);
[~,S2] = ismember(S2, S_pruned);
[~,S3] = ismember(S3, S_pruned);
[~,S4] = ismember(S4, S_pruned);

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
% st:  sum (|a_i^1|, ... , |a_i^d|) < L
% z = [b, a, t, L, epsilon], b in R^n, a in R^(nd),
% t in R^nd,  L in R, epsilon in R^m

I_n_p = eye(K);

% convexity inequalities
A1 = zeros([n_S*(K-1), K*(2*dim+1)+m]);
count = 0;
for i=1:n_S
    for j=setdiff(1:K, c_ind(i))
    % condition that D(i,j)>0
    a = I_n_p(c_ind(i),:) - I_n_p(j,:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((j-1)*dim+1:j*dim) = -X(i,:);
    count = count + 1;
    A1(count,:) = -[a,b,zeros([1,K*dim]),zeros(1,m)];
    end
end
b1 = zeros(n_S*(K-1),1);

% supervision inequalities
A2 = zeros([m, K*(2*dim+1)+m]);
for count=1:m
    i = S1(count); j=S2(count); k=S3(count); l=S4(count);
    % D(i,j)
    a = I_n_p(c_ind(i),:) - I_n_p(c_ind(j),:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((c_ind(j)-1)*dim+1:c_ind(j)*dim) = -X(i,:);
    a1 = [a,b,zeros([1,K*dim]),zeros(1,m)];
    
    % D(k,l)
    a = I_n_p(c_ind(k),:) - I_n_p(c_ind(l),:);
    b = zeros([1,K*dim]);
    b((c_ind(k)-1)*dim+1:c_ind(k)*dim) = X(k,:);
    b((c_ind(l)-1)*dim+1:c_ind(l)*dim) = -X(k,:);
    a2 = [a,b,zeros([1,K*dim]),zeros(1,m)];
    
    % condition that D(k,l) - D(i,j) > 1-eps_ijkl 
    A2(count,:) = a1 - a2;
    A2(count,K*(2*dim+1)+count) = -1;
end
b2 = -1*ones(m,1);

% condition -t_ij < a_ij < t_ij
A3 = [zeros(K*dim,K), -eye(K*dim),-eye(K*dim),zeros(K*dim,m)];
A4 = [zeros(K*dim,K), eye(K*dim),-eye(K*dim),zeros(K*dim,m)];
b3 = zeros(K*dim,1);
b4 = zeros(K*dim,1);

% condition  sum_{j=1}^dim t_ij < L
A5 = [zeros(K,K), zeros(K,2*K*dim), zeros(K,m)];
for i=1:K
    A5(i, K*(dim+1)+(i-1)*dim +1:K*(dim+1)+i*dim) = 1;
end
b5 = L*ones(K,1);
 
% condition that eps_i  > 0
A6 = [zeros(m,K*(2*dim+1)), -eye(m)];
b6 = zeros(m,1);

% price vectors
C = [zeros(1,K*(2*dim+1)),ones(1,m)];                                    % sum (eps_i)

%% solving the LP
A = [A1;A2;A3;A4;A5;A6];
b = [b1;b2;b3;b4;b5;b6];
options = optimoptions('linprog','Algorithm','dual-simplex' ,'Display','final');
z = linprog(C,A,b,[],[],[],[],options);
params.phi = z(1:K);
params.grad = reshape(z(K+1:K*(dim+1)),[dim,K])';

end