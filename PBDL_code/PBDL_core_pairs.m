function params = PBDL_core_pairs(y, X, lambda)
% this is the core code for learning a bregman divergence from pairwise
% inequalities.

% Input: y, X = training data.
% lambda = the hyperparameter of the learning

% Output: params.phi = biases of the learned convex function
% params.grad = slopes of the learned convex function

%% removing the unused training data and partitioning       
[n, d] = size(X);

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


%% v-loss part of constraints
I = zeros(n*(n-1)*2,1);
J = zeros(n*(n-1)*2,1);
V = zeros(n*(n-1)*2,1);


row = 0;
count = 0;
for i=1 : n
    for j=setdiff(1 : n, i)
        row = row + 1;
        count = count + 1;
        iota = 2*(y(i)==y(j))-1;
        
        I(count) = row;
        J(count) = i;
        V(count) = 1*iota;
        
        count = count + 1;
        I(count) = row;
        J(count) = j;
        V(count) = -1*iota;
    end
end
A_v_l = sparse(I,J,V,n*(n-1), n);

%% g-loss part of constraints
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
            iota = 2*(y(i)==y(j))-1;
            I(count) = row;
            J(count) = (j-1)*d + dd;
            V(count) = iota*(- X(i, dd) + X(j, dd));
        end
    end
end
A_g_l = sparse(I,J,V, n*(n-1), n*d);

%% b-loss part of constraints
b3 = zeros(n*(n-1),1);
count = 0;
for i=1 : n
    for j=setdiff(1 : n,i)
        count = count + 1;
        iota = 2*(y(i)==y(j))-1;
        b3(count) = iota - 1;
    end
end

%% eps constraints
I = 1:n*(n-1);
J = 1:n*(n-1);
V = ones(1,n*(n-1));
A_eps = sparse(I,J,V,n*(n-1), n*(n-1));


%% building the constraints
% A1  :  b_i - b_j - (a^+_i - a^-_i)_^T (x_i - x_j) < 0
% A2  :  ||a^+_i|| + ||a^-_i|| + ||b^+_i|| + ||b^-_i|| - L < 0
% A3  :  iota_ij * (b_i - b_j - (a^+_i - a^-_i)_^T (x_i - x_j)) - eps_ij <
% iota_ij - 1


b1 = sparse(n*(n-1),1);
b2 = sparse(n,1);


A1 = [sparse(n*(n-1),n*(n-1)), A_v, A_g, -A_g, sparse(n*(n-1),1)];
A2 = [sparse(n,n+n*(n-1)),A_norm, A_norm, -ones(n,1)];
A3 = [-A_eps, A_v_l, A_g_l, -A_g_l, sparse(n*(n-1),1)];
    

%% cost vector;
c = zeros(1, n + n*(n-1) + 2*n*d + 1);
c(1:n*(n-1)) = 1/(n*(n-1));
c(end) = 1/sqrt(n*(n-1))*lambda;


%% solution bounds
lb = zeros(n + n*(n-1) + 2*n*d + 1, 1);

clearvars A_eps A_g A_v A_g_l A_v_l I J V 

model.obj   = c;
model.lb    = lb;
model.A     = [A1;A2;A3];
model.rhs   = full([b1;b2;b3]);
model.sense = repmat('<',size(model.A,1),1);

fprintf("LP built! ")

params.Threads = 16;
params.outputflag = 0; 
% params.Method = 0;
% Solve
% z = linprog(c,[A1;A2;A3],[b1;b2;b3],[],[],lb);
results = gurobi(model,params);
fprintf("---> LP solved with status "+num2str(results.status)+"!\n")

z = results.x;

shift = n + n*(n-1);

params.phi = z(shift-n+1 : shift);
params.grad = reshape( z(shift + 1 : shift + n*d) - z(shift + n*d + 1 :shift + 2*n*d), [d,n])' ;
params.phi = params.phi - dot(params.grad,X,2);

