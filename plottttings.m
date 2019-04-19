    
%% plotting the clustering assingment
    [~, X_pca, ~] = pca(X_test,'NumComponents',2);
    figure()
    for c=1:3
        ind = find(y_hat==c);
        scatter(X_pca(ind,1),X_pca(ind,2));hold on;
    end
    hold off;
    drawnow
    
    if exist('params', 'var')
        plot_phi(y_hat, X_test, params); drawnow
    end
    