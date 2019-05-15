%%
h = figure('Name', 'ali','Position', 1.5*[0 0 550 150]);
zn = 1.96;
num_run = 50;
min_n = 5;
max_n = 100;
n_array = min_n:5:max_n;
n_array = n_array';
breg = ["ItSa", "GID", "mahal"];
col = get(gca,'colororder');

for data = [1,2,3]
    
    subplot (1,3,data); hold on
    load("regression2/NBR_"+breg(data)+".mat");
    err1 = err;
    load("regression2/MR_"+breg(data)+".mat");
    err2 = err;
    clear err
    
    e1 = zn*std(err1,0,2)/sqrt(num_run);
    e2 = zn*std(err2,0,2)/sqrt(num_run);
    
    
    lo1 = max(mean(err1,2) - e1,0);
    hi1 = mean(err1,2) + e1;
    lo2 = max(mean(err2,2) - e2,0);
    hi2 = mean(err2,2) + e2;
    
    hp2 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo2; hi2(end:-1:1); lo2(1)], col(2,:));
    hl2 = plot(n_array,mean(err2,2), '--', 'Color', col(2,:));
    hl2.LineWidth = 2;
    
    hp1 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo1; hi1(end:-1:1); lo1(1)], col(1,:));
    hl1 = plot(n_array,mean(err1,2), 'Color', col(1,:));
    hl1.LineWidth = 2;
    
    set(hp1, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
    set(hp2, 'facecolor', [1.0 0.8 0.8], 'edgecolor', 'none');
    
    switch breg
        case 'ItSa'
            title('Itakura-Saito distance')
        case 'GID'
            title('Generalized I-divergence')
        case 'Mahal'
            title('Mahalanobis distance')     
    end
    xlim([min(n_array) max(n_array)])
    
end


legend([hl1, hl2],["Bregman regression", "Mahalanobis regression"])
legend

%%
h ;
h.NextPlot = 'add';
a = axes;

%// Set the title and get the handle to it
% ht = title('Test');
hx = xlabel('Number of training data');
hy = ylabel('$E \|D_{h_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',14);

%// Turn the visibility of the axes off
a.Visible = 'off';

%// Turn the visibility of the title on
% ht.Visible = 'on';
hx.Visible = 'on';
hy.Visible = 'on';

print -depsc epsFig_reg_all


