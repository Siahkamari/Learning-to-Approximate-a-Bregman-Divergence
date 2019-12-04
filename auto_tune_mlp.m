function net = auto_tune_mlp(y, X, S1, S2)
n_folds = 3;

XS1= X(S1,:);
XS2= X(S2,:);

hidd_array = [5;10;15;30;50;100];
n_hyper_choices = length(hidd_array);
cvErr = zeros(n_hyper_choices,1);

for hyper_choice = 1:n_hyper_choices
    CVO = cvpartition(length(y),'k',n_folds);
    errcv = zeros(CVO.NumTestSets,1);
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        
        net = fitnet(hidd_array(hyper_choice));
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 30/100;
        net.trainParam.showWindow = false;
        
        
        
        [~ , tr] = train(net,[XS1(trIdx,:),XS2(trIdx,:)]',y(trIdx)','useGPU','yes');
        errcv(i) = tr.best_vperf;
    end
    cvErr(hyper_choice) = mean(errcv);
end

[~,i] = min(cvErr);
hiddenLayerSize = hidd_array(i);

net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;

    net.trainParam.showWindow = false;
    net = train(net,[X(S1,:),X(S2,:)]',y','useGPU','yes');
