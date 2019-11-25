function params = PBDL_core(X, S1, S2, S3, S4, lambda)
% this is the core code for learning a bregman divergence from pairwise
% inequalities.

% Input: X_train = training data. S1 and S2 are indices of random chosen
% training data from the same cluster. S3 and S4 are form different
% clusters. 
% lambda = the hyperparameter of the learning

% Output: params.phi = biases of the learned convex function
% params.grad = slopes of the learned convex function


%% removing the unused training data and partitioning       
S_pruned = union(union(union(S1,S2),S3),S4);

[~,S1] = ismember(S1, S_pruned);
[~,S2] = ismember(S2, S_pruned);
[~,S3] = ismember(S3, S_pruned);
[~,S4] = ismember(S4, S_pruned);

X = X(S_pruned,:);
[n, d] = size(X);
m = length(S1);

%% v part of constraints
I = zeros(n*(n-1)*2,1);
J = zeros(n*(n-1)*2,1);
V = zeros(n*(n-1)*2,1);

row = 0;
count = 0;
for i=1 : n
    for j=setdiff(1 : n, i)
        row = row + 1;
        count = count + 1;
        
        I(count) = row;
        J(count) = i;
        V(count) = 1;
        
        count = count + 1;
        I(count) = row;
        J(count) = j;
        V(count) = -1;
    end
end
A_v = sparse(I,J,V,n*(n-1), n);

%% g part of constraints
I = zeros(n*(n-1)*d,1);
J = zeros(n*(n-1)*d,1);
V = zeros(n*(n-1)*d,1);

row = 0;
count = 0;
for i=1 : n
    for j=setdiff(1 : n,i)
        row = row + 1;
        for dd=1:d
            count = count + 1;
            I(count) = row;
            J(count) = (i-1)*d + dd;
            V(count) = - X(i, dd) + X(j, dd);
        end
    end
end
A_g = sparse(I,J,V, n*(n-1), n*d);

%% norm constraints
I = zeros(n*d,1);
J = zeros(n*d,1);
V = ones(n*d,1);
count = 0;
for i = 1:n
    for dd=1:d
        count = count+1;
        I(count) = i;
        J(count) = (i-1)*d + dd;
    end
end
A_norm = sparse(I,J,V, n, n*d);

%% loss constraints
I = zeros(m*(2+2*d),1);
J = zeros(m*(2+2*d),1);
V = zeros(m*(2+2*d),1);

for t = 1:m
    I((t-1)*(2+2*d)+1:t*(2+2*d)) = t;
    J((t-1)*(2+2*d)+1:t*(2+2*d)) = [S1(t),S2(t),...
        n+(S2(t)-1)*d+1:n+S2(t)*d, n+n*d+(S2(t)-1)*d+1:n+n*d+S2(t)*d];
    V((t-1)*(2+2*d)+1:t*(2+2*d)) = [1, -1, -X(S1(t),:) + X(S2(t),:), X(S1(t),:) - X(S2(t),:)];
end
A_dx1x2 = sparse(I,J,V, m, n + 2*n*d);

I = zeros(m*(2+2*d),1);
J = zeros(m*(2+2*d),1);
V = zeros(m*(2+2*d),1);

for t = 1:m
    I((t-1)*(2+2*d)+1:t*(2+2*d)) = t;
    J((t-1)*(2+2*d)+1:t*(2+2*d)) = [S3(t),S4(t),...
        n+(S4(t)-1)*d+1:n+S4(t)*d, n+n*d+(S4(t)-1)*d+1:n+n*d+S4(t)*d];
    V((t-1)*(2+2*d)+1:t*(2+2*d)) = [1, -1, -X(S3(t),:) + X(S4(t),:), X(S3(t),:) - X(S4(t),:)];
end
A_dx3x4 = sparse(I,J,V, m, n + 2*n*d);
A_loss = sparse(1:m,1:m,-ones(m,1), m, m);

%% building the constraints
% A1  :  b_i - b_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0
% A3  :  b_i - b_j - (a^+_i - a^-_i)_^T (x_i - x_j)
%            -[b_k - b_l - (a^+_k - a^-_k)_^T (x_k - x_l)] - eps_t < -1

b1 = sparse(n*(n-1),1);
b2 = sparse(n,1);
b3 = -ones(m,1);

if lambda > 0
    A1 = [sparse(n*(n-1),m), A_v, A_g, -A_g, sparse(n*(n-1),1)];
    A2 = [sparse(n,n+m),A_norm,A_norm, -ones(n,1)];
    A3 = [A_loss, A_dx1x2 - A_dx3x4, sparse(m,1)];
    
elseif lambda == 0 
    A1 = [A_v, A_g, -A_g, sparse(n*(n-1),1)];
    A2 = [sparse(n,n),A_norm,A_norm, -ones(n,1)];
    A3 = [A_dx1x2 - A_dx3x4, sparse(m,1)];
end

%% cost vector;
if lambda > 0
    c = zeros(1, n + m + 2*n*d + 1);
    c(1:m) = 1;
    c(end) = lambda;    
elseif lambda == 0 
    c = zeros(1, n + 2*n*d + 1);
    c(end) = 1;
end

%% solution bounds
if lambda > 0
    lb = zeros(n + m + 2*n*d + 1, 1);
elseif lambda == 0 
    lb = zeros(n + 2*n*d + 1, 1);
end

%% solving
clearvars A_dx1x2 A_dx3x4 A_g A_v A_loss A_norm I J V S1 S2 S3 S4 S_pruned
options = optimoptions('linprog','Display', 'iter');
try
    z = linprog_gurobi(c,[A1;A2;A3],[b1;b2;b3],[],[],lb,[],[],options);
catch
    warning('Gurobi is not installed/working, trying Matlab solvers instead');
    z = linprog(c,[A1;A2;A3],[b1;b2;b3],[],[],lb,[],options);
end

if lambda > 0
    shift = n + m;
elseif lambda == 0
    shift = n;
end

params.phi = z(shift-n+1 : shift);
params.grad = reshape( z(shift + 1 : shift + n*d) - z(shift + n*d + 1 :shift + 2*n*d), [d,n])' ;
params.phi = params.phi - dot(params.grad,X,2);

