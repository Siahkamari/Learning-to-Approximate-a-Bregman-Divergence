function [S1, S2, S3, S4] = get_supervision_pairs(m, y_train)
% Input: m number of supervisions needed. y_train = class labels 
% Output: S1 and S2 index of randomly chosen pairs that are in the same
% cluster

%% computing the data parameters from data


n_train = length(y_train);

classes = unique(y_train);
c = length(classes);

n_train_c = zeros(c,1);

inds = cell(c,1);
for i=1:c
    inds{i} = find(y_train == classes(i));
    n_train_c(i) = length(inds{i});
end                                       
                                                          
%% supervision points
S1 = zeros(m,1);
S2 = zeros(m,1);
S4 = zeros(m,1);

pi = makedist('Multinomial','probabilities',n_train_c/n_train);

for count = 1:m

    i = random(pi);
    j = i;
    while j==i
        j = random(pi);
    end
    
    S1(count) = randsample(inds{i},1); 
    S2(count) = randsample(inds{i},1); 
    S4(count) = randsample(inds{j},1); 
end
S3 = S1;
