function plot_phi(y, X, params)

d_mesh = 0.1;                                                               % mesh increments

dim = size(X,2);
if isa(params, 'double')
    phi_X = dot(X*params, X, 2);
elseif isa(params, 'struct')
    phi_X = max(params.phi' + X*params.grad', [], 2);
end


max_y = max(y);
ind = cell(max_y,1);
for i=1:max_y
    ind{i} = find(y==i);
end

if dim==1                                                                   % plotting the points X
    for i=1:max_y
        scatter(X(ind{i}),phi_X(ind{i}), 'filled');hold on
    end
    xlabel('x');ylabel('\phi(x)')
else
    for i=1:max_y
        scatter3(X(ind{i},1),X(ind{i},2),phi_X(ind{i}),'filled');hold on
    end
    xlabel('x_1');ylabel('x_2');zlabel('\phi(x)');grid on
end

if dim==1                                                                   % plotting the mesh surface
    X_mesh = (min(X(:,1)):d_mesh:max(X(:,1)))';
    if isa(params, 'double')
        phi_mesh = dot(X_mesh*params, X_mesh, 2);
    elseif isa(params, 'struct')
        phi_mesh = max(params.phi' + X_mesh*params.grad', [], 2);
    end
    plot(X_mesh,phi_mesh)
    xlabel('x');ylabel('\phi(x)')
    colormap(gray)
else
    alpha = 0.3;
    [X1mesh,X2mesh] = meshgrid(min(X(:,1))-alpha:d_mesh:max(X(:,1))+alpha,...
        min(X(:,2))-alpha:d_mesh:max(X(:,2))+alpha);
    X_mesh = [X1mesh(:),X2mesh(:)];  
    for i=3:dim
        X_mesh(:,i) = mean(X(:,i));                                         % 3+ dimmension of mesh = mean(X(:,dim));
    end
    
    if isa(params, 'double')
        phi_mesh = dot(X_mesh*params, X_mesh, 2);
    elseif isa(params, 'struct')
        phi_mesh = max(params.phi' + X_mesh*params.grad', [], 2);
    end
    
    X3mesh = reshape(phi_mesh, size(X1mesh));
    meshc(X1mesh,X2mesh,X3mesh)
    xlabel('x_1'); ylabel('x_2'); zlabel('\phi(x)'); grid on
    colormap(gray)
end

hold off