function A = MR(y, X_train, S1, S2)
% Mahalanobis regression

[~, dim] = size(X_train);

%% algorithm parameters

S_pruned = union(S1,S2);

[~,S1] = ismember(S1,S_pruned);
[~,S2] = ismember(S2,S_pruned);

%% removing the unused training data and partitioning
X = X_train(S_pruned,:); % if data already exists

%% preparing the optimization loss and grad
% min 1/n sum(xi' A xi - yi)^2

options = optimoptions('fminunc','SpecifyObjectiveGradient',true, 'Display', 'off');
A0 = eye(dim);
fun = @(A)mahal_loss(y,X,S1,S2,A);
A = fminunc(fun,A0(:),options);
A = reshape(A, [size(X,2),size(X,2)]);

    function [loss, grad] = mahal_loss(y,X,S1,S2,A) 
        A = reshape(A, [size(X,2),size(X,2)]);
        
        n = length(y);
        loss = 0.5/n*sum((dot( (X(S1,:)-X(S2,:))*A,  X(S1,:)-X(S2,:), 2)-y).^2);
        grad = 0;
        for i=1:n
            
            grad = grad + 1/n * ((X(S1(i),:)-X(S2(i),:))*A*(X(S1(i),:)-X(S2(i),:))' - y(i))*...
                ((X(S1(i),:)-X(S2(i),:))'*(X(S1(i),:)-X(S2(i),:)));
            
        end
        grad = grad(:);
    end
end