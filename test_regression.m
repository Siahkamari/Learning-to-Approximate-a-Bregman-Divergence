clear, clc
rng(0);

%% experiment setting
dset = 'unit_box';
breg = 'ItSa';
% breg = 'Mahal';
% breg = 'GID';
% method = 'NBR';
% method = 'MR';
method = 'MLP';
L = 100;
min_n = 5;
max_n = 100;
n_test = 1000;
sigma = 0.05;
dim = 2;
num_run = 20;
n_array = (min_n:5:max_n);

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
            Ey_test = mahalanobis(X_test(S1_test,:), X_test(S2_test,:), A);
            Ey = mahalanobis(X(S1,:), X(S2,:), A);
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
                bregman_div = @(X1,X2)max_affine_bregman(X1,X2,params);
            case "MR"
                A = MR(y, X, S1, S2);
                bregman_div =  @(X1,X2)mahalanobis(X1,X2,A);
            case "MLP"
                hiddenLayerSize = 10;
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

% save('NBR_mahal.mat','err')

n_array = n_array';

%% Plotting
figure('Position', [0 0 350 300]);  hold on
e = std(err,0,2)/sqrt(num_run);
lo = max(0,mean(err,2) - 2*e);
hi = mean(err,2) + 2*e;
hp = patch([n_array; n_array(end:-1:1);n_array(1)], [lo; hi(end:-1:1); lo(1)], 'b');
alpha(0.8);
hl = plot(n_array,mean(err,2));
hl.LineWidth = 2;
set(hp, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
set(hl, 'color', 'b');
xlabel("number of data points")
xlim([min_n,max_n])
ylabel('$ E \|D_{\phi_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',12)
switch breg
    case 'ItSa'
        title('Itakura-Saito distance')
    case 'Mahal'
        title('Mahalanobis distance')
    case 'GID'
        title('Generalized I-divergence')   
end
switch method
    case 'NBR'
        legend('Bregman regression')
    case 'MR'
        legend('Mahalanobis regression')
    case 'MLP'
        legend('MLP regression')   
end


%% local functions
function y = it_sa_dist(X1,X2)
y = sum(X1./X2 -log(X1./X2)-1,2);
end
function y = general_i_div(X1,X2)
y = sum(X1.*log(X1./X2) - (X1-X2),2);
end