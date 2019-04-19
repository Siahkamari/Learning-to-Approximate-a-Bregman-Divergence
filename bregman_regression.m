clear, clc
%% parameters
n_train = 50;                  % number of points in training
n_test = 1000;                  % number of points in test
dim = 2;                        % dimension of data
m = 5000;                       % number of supervision
noise = 0.05;                   % noise of supervision
lambda = 1;                  % regularization
d_mesh = 0.1;                   % mesh distanse for plotting
type = "kl";

%% Supervision points
S1 = randsample(n_train,m,true);                  % sampling with replacement
S2 = randsample(n_train,m,true);
S_pruned = union(S1,S2);

[~,S1] = ismember(S1,S_pruned);
[~,S2] = ismember(S2,S_pruned);

% X_pruned = X(I_pruned);                         % if data already exists
n_S = length(S_pruned);

%% getting data
out = get_data(n_S, n_test, dim, type, noise);
X = out{1}; X_test = out{2}; dis_noised = out{3}; dis_test = out{4};

%% preparing the optimization constants
% min ||C*z-d|| + lambda ||g||, z = [y,g], y in R^n, g in R^(nd)
% ~ min ||Cprim*z - dprim||

Cprim = zeros([m+1, n_S*(dim+1)]);
dprim = zeros([m+1,1]);
I = eye(n_S);
for count=1:m
    i = S1(count); j=S2(count);
    dprim(count) = dis_noised(i,j);
    a = I(i,:) - I(j,:);
    b = zeros([1,n_S*dim]);
    b((j-1)*dim+1:j*dim) = X(j,:)-X(i,:);
    Cprim(count,:) = [a,b];
end

Cprim(end,n_S+1:end) = sqrt(lambda);      % adding the regularization term
dprim(end) = 0;

A = zeros([n_S^2, n_S*(dim+1)]);
for i=1:n_S
    for j=1:n_S
    % Condition that D(i,j)>0
    a = I(i,:) - I(j,:);
    b = zeros([1,n_S*dim]);
    b((j-1)*dim+1:j*dim) = X(j,:)-X(i,:);
    A(i+(j-1)*n_S,:) = [a,b];
    end
end
b = zeros([n_S^2,1]);

%% solving constrained linear least-squares
[z,resnorm,residual,exitflag,output,lambda] = lsqlin(Cprim,dprim,-A,b);
y = z(1:n_S);
g = reshape(z(n_S+1:end),[dim,n_S])';

%% predition
phi_hat = zeros([n_test,1]);
index = zeros([n_test,1]);
for i=1:n_test
    [phi_hat(i),index(i)] = max(y + sum(g.*(X_test(i,:)-X), 2));
end
grad_hat = g(index,:);

dis_hat = zeros([n_test,n_test]);
for i=1:n_test
    for j=1:n_test
        dis_hat(i,j) = phi_hat(i)-phi_hat(j) - (X_test(i,:)-X_test(j,:))*grad_hat(j,:)';
    end
end
test_eror = sum(sum((dis_test-dis_hat).^2))/sum(sum((dis_test.^2)));


%% plotting
figure(1)
plot(dis_test(:),dis_hat(:),'.')
xlabel('true divergence')
ylabel('predictd divergence')

figure(2)
if dim==1
    plot(X_test,phi_hat,'.');hold on
    xlabel('x');ylabel('\phi(x)')
elseif dim==2
    plot3(X_test(:,1),X_test(:,2),phi_hat,'.'); hold on
    xlabel('x_1');ylabel('x_2');zlabel('\phi(x)');grid on
end

figure(2)
if dim==1
    X_mesh = (min(X(:,1)):d_mesh:max(X(:,1)))';
    n_Xmesh = size(X_mesh,1);
    phi_hat_mesh = zeros([n_Xmesh, 1]);
    for i=1:n_Xmesh
        phi_hat_mesh(i) = max(y + sum(g.*(X_mesh(i,:)-X), 2));
    end
    plot(X_mesh,phi_hat_mesh)
    xlabel('x');ylabel('\phi(x)')
elseif dim==2
    [X1mesh,X2mesh] = meshgrid(min(X(:,1)):d_mesh:max(X(:,1)),min(X(:,2)):d_mesh:max(X(:,2)));
    X_mesh = [X1mesh(:),X2mesh(:)];
    n_Xmesh = size(X_mesh,1);
    phi_hat_mesh = zeros([n_Xmesh, 1]);
    for i=1:n_Xmesh
        phi_hat_mesh(i) = max(y + sum(g.*(X_mesh(i,:)-X), 2));
    end
    X3mesh = reshape(phi_hat_mesh, size(X1mesh));
    mesh(X1mesh,X2mesh,X3mesh)
    xlabel('x_1');ylabel('x_2');zlabel('\phi(x)');grid on
end