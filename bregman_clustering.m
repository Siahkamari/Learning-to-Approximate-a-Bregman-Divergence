function y_hat = bregman_clustering(X_test, num_cluster, bregman_div)

[n_test, dim] = size(X_test);
% y_hat = randi(num_cluster,[n_test,1]);                                    % initial assingment
y_hat = kmeans(X_test, num_cluster);

converge = false;
for iter=1:400
    X_bar = zeros(num_cluster,dim);                                         % M step
    for i=1:num_cluster
        X_bar(i,:) = mean(X_test(y_hat==i,:));
    end
    
    divs = bregman_div(X_test, X_bar);                               % computing divergences
    
    [~, y_hat_new] = min(divs, [], 2);                                      % assignment step
   
    if sum(y_hat_new~=y_hat)==0                                             % loop termination check
        converge = true;
        break
    else
        y_hat = y_hat_new;
    end
    
end

if ~converge
    disp('Warning: bregman clustering didnt converge');
end