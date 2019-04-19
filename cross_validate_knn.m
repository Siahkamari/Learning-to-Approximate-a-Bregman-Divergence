function [acc, pred] = cross_validate_knn(y, X, tCL, num_folds, knn_size, method)

if min(y) == 0
    y = y + 1;
end

[n,dim] = size(X);
if (n ~= length(y))
    disp('ERROR: num rows of X must equal length of y');
    return;
end

%permute the rows of X and y
rp = randperm(n);
y = y(rp);
X = X(rp, :);

pred = zeros(n,1);
for i=1:num_folds
    test_start = ceil(n/num_folds * (i-1)) + 1;
    test_end = ceil(n/num_folds * i);
    
    y_train = [];
    X_train = [];
    if i > 1
        y_train = y(1:test_start-1);
        X_train = X(1:test_start-1,:);
    end
    if i < num_folds
        y_train = [y_train; y(test_end+1:length(y))];
        X_train = [X_train; X(test_end+1:length(y), :)];
    end
    X_test = X(test_start:test_end, :);
    
    if method == "Euclidean"
        M = eye(dim);
        divs = mahalanobis(X_test, X_train, M);
    elseif method == "ITML"
        M = feval(tCL, y_train, X_train);
        divs = mahalanobis(X_test, X_train, M);
    elseif method == "NBDL"
        m = 1000;
        [S1, S2, S3, S4] = get_supervision_pairs(m, y_train);
        params = NBDL(X_train, S1, S2, S3, S4);
        divs = max_affine_bregman(X_test, X_train, params);
    end
    
    pred(test_start:test_end) = divergence_knn(y_train, divs, knn_size);
    
%     if exist('params', 'var')
%         plot_phi(pred(test_start:test_end), X_test, params); drawnow
%     end
end
acc = sum(pred==y)/n;
