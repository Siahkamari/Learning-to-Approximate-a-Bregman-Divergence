function [S1, S2, S3, S4] = get_supervision_pairs(m, y_train)
                                                                % number of supervision                                                             % noise of supervision                                                                                                                             % trade-off of bias/variance                                                           % number of hyper-planes

%% computing the data parameters from data
max_y = max(y_train);
n_train = length(y_train);
n_train_e = zeros(max_y,1);
inds = cell(max_y,1);
nempty_class = 0;
for i=1:max_y
    n_train_e(i) = sum(y_train==i);
    inds{i} = find(y_train == i);
    if ~isempty(inds{i})
        nempty_class = nempty_class +1 ;
    end
end                                       
                                                          
%% supervision points
m_e = floor(m/(nempty_class*(nempty_class-1)));
m = nempty_class*(nempty_class-1)*m_e;
S1 = zeros(m,1);
S2 = zeros(m,1);
S4 = zeros(m,1);

% if n_train > 400
%     for i=1:max_y
%         inds{i} = inds{i}(inds{i}<200);
%     end
% end

count = 0;
for i = 1:max_y
    if isempty(inds{i})
        continue
    end
    for j=setdiff(1:max_y,i)
        if isempty(inds{j})
            continue
        end
        S1(count+1:count+m_e) = randsample(inds{i},m_e,true); 
        S2(count+1:count+m_e) = randsample(inds{i},m_e,true); 
        S4(count+1:count+m_e) = randsample(inds{j},m_e,true); 
        count = count + m_e;
    end
end
S3 = S1;
