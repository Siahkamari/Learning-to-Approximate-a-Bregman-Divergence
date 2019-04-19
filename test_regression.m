clear, clc
rng(0);

%% experiment setting
dset = 'unit_box';
% breg = 'ItSa';
% breg = 'Mahal';
breg = 'GID';
method = 'NBR';
% method = 'MR';
% method = 'MLP';
L = 10^10;
max_n = 50;
n_test = 1000;
sigma = 0.05;
dim = 1;
num_run = 1;
n_array = 10:5:max_n;

%% running
err = zeros(length(n_array),num_run);
for run=1:num_run
    clc
    fprintf('data-set %d/%d \n',run,num_run)
    switch dset
        case 'unit_box' % box of 0.1 to 1.1
            X = rand(max_n,dim) + 0.1;
            X_test = rand(n_test,dim) + 0.1;
    end
    
    [S1_test,S2_test] = meshgrid(1:n_test,1:n_test);
    S1_test = S1_test(:); S2_test = S2_test(:);
    [S1,S2] = meshgrid(1:max_n,1:max_n);
    S1 = S1(:); S2 = S2(:);
    
    switch breg
        case 'ItSa'
            Ey_test = it_sa_dist(X_test(S1_test,:),X_test(S2_test,:));
            Ey = it_sa_dist(X(S1,:),X(S2,:));
        case 'Mahal'
            A = eye(dim) + 0.5*ones(dim);
            Ey_test = mahalanobis_notall(X_test(S1_test,:), X_test(S2_test,:), A);
            Ey = mahalanobis_notall(X(S1,:), X(S2,:), A);
        case 'GID'
            Ey_test = general_i_div(X_test(S1_test,:), X_test(S2_test,:));
            Ey = general_i_div(X(S1,:), X(S2,:));
    end
    Ey = reshape(Ey, [max_n, max_n])';
    noise = sigma*randn(max_n,max_n);
    y_all = Ey + noise;
    
    counter = 0;
    for n=n_array
        fprintf('training on %d/%d data points \n',n,max_n)
        counter = counter + 1;
        %% supervision pairs
        [S1,S2] = meshgrid(1:n,1:n);
        S1 = S1(:);
        S2 = S2(:);
        
        %% computing distances
        y = y_all(sub2ind(size(y_all),S1,S2));
        
        switch method
            case "NBR"
                out = NBR(y, X, S1, S2, L);
                params = out{1};
                bregman_div = @(X1,X2)max_affine_bregman(X1,X2,params,'not');
            case "MR"
                A = MR(y, X, S1, S2);
                bregman_div =  @(X1,X2)mahalanobis(X1,X2,A,'not');
            case "MLP"
                hiddenLayerSize = 5;
                net = fitnet(hiddenLayerSize);
                net.divideParam.trainRatio = 80/100;
                net.divideParam.valRatio = 20/100;
                [net,tr] = train(net,[X(S1,:),X(S2,:)]',y');
                bregman_div =  @(X1,X2)net([X1,X2]')';
        end
        
        y_hat = bregman_div(X_test(S1_test,:),X_test(S2_test,:));
        
        err(counter,run) = 1/n_test^2 * norm(y_hat-Ey_test)^2;
    end
end
h = plot(n_array,mean(err,2)); hold on
h.LineWidth = 2;
legend( "Bregman regression","Mahalanobis regression", "MlP regression")
xlabel("number of data points")
ylabel('E \|D_{\phi _n}- D_{\phi_*} \|')
switch breg
    case 'ItSa'
        title('Itakura-Saito distance')
    case 'Mahal'
        title('Mahalanobis distance')
    case 'GID'
        title('General I divergence')   
end

function y = it_sa_dist(X1,X2)
y = sum(X1./X2 -log(X1./X2)-1,2);
end
function y = general_i_div(X1,X2)
y = sum(X1.*log(X1./X2) - (X1-X2),2);
end