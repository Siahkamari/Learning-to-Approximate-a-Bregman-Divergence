clear, clc
rng(0);

%% experiment setting
dset = 'unit_box';
dset = 'drch_probs';
dset = 'herm_matrix';

breg = 'ItSa';
breg = 'Mahal';
breg = 'GID';
breg = 'KL';
breg = 'LogDet';

method = 'NBR';
% method = 'MR';
% method = 'MLP';

L = 100;
min_n = 5;
max_n = 100;
n_test = 1000;
sigma = 0.05;
dim = 2;
num_run = 50;
n_array = (min_n:5:max_n);

%% running
err = zeros(length(n_array),num_run);
for run=1:num_run
    clc
    fprintf('run %d/%d \n',run,num_run)
    switch dset
        case 'unit_box' % box of 0.1 to 1.1
            X = rand(max_n,dim) + 0.1;
            X_test = rand(n_test,dim) + 0.1;
        case 'drch_probs'
            X = drchrnd(max_n, ones([1, dim]));
            X_test = drchrnd(n_test, ones([1, dim]));
        case 'herm_matrix'
            X = matrnd(max_n, dim);
            X_test = matrnd(n_test, dim);  
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
        case 'KL'
            Ey_test = kl_divergence(X_test(S1_test,:), X_test(S2_test,:));
            Ey = kl_divergence(X(S1,:), X(S2,:));
        case 'LogDet'
            Ey_test = ld_divergence(X_test(S1_test,:), X_test(S2_test,:));
            Ey = ld_divergence(X(S1,:), X(S2,:));
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
                hiddenLayerSize =5;
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

save(method+"_"+breg+".mat",'err')
n_array = n_array';

%% Plotting
zn = 1.96;
% figure('Position', [0 0 350 300]);  hold on
e = std(err,0,2)/sqrt(num_run);
lo = max(0,mean(err,2) - zn*e);
hi = mean(err,2) + zn*e;
hp = patch([n_array; n_array(end:-1:1);n_array(1)], [lo; hi(end:-1:1); lo(1)], 'b');
set(hp, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
alpha(0.8);
hl = plot(n_array,mean(err,2));
hl.LineWidth = 2;
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

function y = kl_divergence(X1,X2)
y = sum(X1.*log(X1./X2),2);
end

function y = ld_divergence(X1,X2)
[n, dim] = size(X1);
dim = sqrt(dim);
y = zeros(n,1);
for i=1:n
    X = reshape(X1(i,:), [dim, dim]);
    Y = reshape(X2(i,:), [dim, dim]);
    T = X/Y ;
    y(i) = trace(T) - log(det(T)) - dim;
end
end

function r = drchrnd(n,a)
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end

function r = matrnd(n,dim)
r = zeros(n,dim^2);
for i=1:n
    r(i,:) = reshape(wishrnd(eye(dim),10)/10, [], 1);
end
end
