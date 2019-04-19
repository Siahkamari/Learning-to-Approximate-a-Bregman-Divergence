function params = NBDLLinf_sum(X_train, S1, S2, S3, S4)

[n_train, dim] = size(X_train);


%% algorithm parameters
% K = ceil(n_train^(dim/(dim+4)));
K = 100;
lambda = 100;  
m = length(S1);  

S_pruned = union(union(union(S1,S2),S3),S4);

[~,S1] = ismember(S1,S_pruned);
[~,S2] = ismember(S2,S_pruned);
[~,S3] = ismember(S3,S_pruned);
[~,S4] = ismember(S4,S_pruned);

%% removing the unused training data and partitioning 
X = X_train(S_pruned,:); % if data already exists
n_S = length(S_pruned);
if K < n_S
    [~, c_ind] = farthest_point_clustering(X,K);                       % finding the partions
else
    K = n_S; c_ind = (1:n_S)';
end

%% preparing the optimization constants
% min sum(t) + eta* sum(epsilon)
% st: D(i,j) - D(i,k) > 1 - epsilon_ijk
% st: D(i,j) >= 0
% st: epsilon_ijk >= 0
% st: -t_i < g_i^1, ... , g_i^d < t_i
% z = [y,g, t, epsilon], y in R^n, g in R^(nd),
% t in R^n, epsilon in R^m

I_n_p = eye(K);

% convexity inequalities
A1 = zeros([n_S*(K-1), K*(dim+2)+m]);
count = 0;
for i=1:n_S
    for j=setdiff(1:K, c_ind(i))
    % condition that D(i,j)>0
    a = I_n_p(c_ind(i),:) - I_n_p(j,:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((j-1)*dim+1:j*dim) = -X(i,:);
    c = zeros(1,K);
    count = count + 1;
    A1(count,:) = [a,b,c,zeros(1,m)];
    end
end

% supervision inequalities
A3 = zeros([m, K*(dim+2)+m]);
for count=1:m
    i = S1(count); j=S2(count); k=S3(count); l=S4(count);
    % D(i,j)
    a = I_n_p(c_ind(i),:) - I_n_p(c_ind(j),:);
    b = zeros([1,K*dim]);
    b((c_ind(i)-1)*dim+1:c_ind(i)*dim) = X(i,:);
    b((c_ind(j)-1)*dim+1:c_ind(j)*dim) = -X(i,:);
    c = zeros(1,K);
    a1 = [a,b,c,zeros(1,m)];
    
    % D(k,l)
    a = I_n_p(c_ind(k),:) - I_n_p(c_ind(l),:);
    b = zeros([1,K*dim]);
    b((c_ind(k)-1)*dim+1:c_ind(k)*dim) = X(k,:);
    b((c_ind(l)-1)*dim+1:c_ind(l)*dim) = -X(k,:);
    c = zeros(1,K);
    a2 = [a,b,c,zeros(1,m)];
    
    % condition that D(k,l) - D(i,j) > 1-eps_ijkl 
    A3(count,:) = -1*(a1 - a2);
    A3(count,K*(dim+2)+count) = 1;
end

% condition -t_i < g_ij < t_i
A4 = [zeros(K*dim,K), eye(K*dim),zeros(K*dim,K),zeros(K*dim,m)];
A5 = A4;
for i=1:K
    A4((i-1)*dim+1:i*dim,K*(dim+1)+i) = 1;
    A5((i-1)*dim+1:i*dim,K*(dim+1)+i) = -1;
end

% condition that eps_i  > 0
A6 = [zeros(m,K*(dim+2)), eye(m)];

% price vectors
C1 = [zeros(1,K*(dim+1)),ones(1,K),zeros(1,m)];                         % sum (t_i)
C2 = [zeros(1,K*(dim+1)),zeros(1,K),ones(1,m)];                         % sum (eps_i)
C = C1 + lambda*C2;

%% solving the LP
A = [-A1;-A3;-A4;A5;-A6];
b = [zeros(n_S*(K-1),1);-1*ones(m,1);zeros(2*K*dim,1);zeros(m,1)];
options = optimoptions('linprog','Algorithm','interior-point','Display','iter');
z = linprog(C,A,b,[],[],[],[],options);
params.phi = z(1:K);
params.grad = reshape(z(K+1:K*(dim+1)),[dim,K])';

end