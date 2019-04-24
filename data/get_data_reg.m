function out = get_data(n_S, n_test, dim, type, noise)


%% data and distance generation - any data X in R^(n*d) can be imported

if type == "mahal"
    X = [randn([n_S/2,dim]);randn([n_S/2,dim])];
    X_test = [randn([n_test/2,dim]);randn([n_test/2,dim])]; 
    % % mahalanobis
    rng(0)
    % A = wishrnd(eye(dim),100)/100
    A = eye(dim);
    % A = [1.2, 0.1; 0.1, 1];
    dis = mahalanobis(X,A);
    dis_test = mahalanobis(X_test,A);
    
elseif type == "kl"
    % probability data
    X = drchrnd(ones([1, dim]),n_S);
    X_test = drchrnd(ones([1, dim]),n_test);
    
    dis = kl_divergence(X);
    dis_test = kl_divergence(X_test);
end
dis_noised = dis + noise*randn([n_S,n_S]);


%% distance functions
    function dis = mahalanobis(X,A)
        L = chol(A);
        n = size(X,1);
        deltaX = permute(repmat(X*L, [1, 1, n]),[1,3,2]);
        deltaX = (deltaX - permute(deltaX,[2,1,3]));
        dis = deltaX.^2;
        dis = sum(dis,3);
    end

    function dis = kl_divergence(X)
        n = size(X,1);
        dis = zeros([n,n]);
        for i=1:n
            dis(:,i) = sum(X.*log(X./X(i,:)),2);
        end
    end

    function r = drchrnd(a,n)
        p = length(a);
        r = gamrnd(repmat(a,n,1),1,n,p);
        r = r ./ repmat(sum(r,2),1,p);
    end

out = {X, X_test, dis_noised, dis_test};

end
