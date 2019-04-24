%%
h = figure('Name', 'ali','Position', 1.5*[0 0 550 150]);

%% Itsa

subplot 131; hold on
load('regression/NBR_IT.mat');
err1 = err;
load('regression/MR_IT.mat');
err2 = err;
load('regression/MLP_IT.mat');
err3 = err(2:20,:);

clear err

breg = 'ItSa';
% breg = 'Mahal';
% breg = 'GID';
num_run = 20;
min_n = 10;
max_n = 100;
n_array = (min_n:5:max_n);


n_array = n_array';
%% ItSa
% figure('Position', [0 0 350 300]);  hold on
col = get(gca,'colororder');

e1 = std(err1,0,2)/sqrt(num_run);
e2 = std(err2,0,2)/sqrt(num_run);
e3 = std(err3,0,2)/sqrt(num_run);


lo1 = max(mean(err1,2) - e1,0);
hi1 = mean(err1,2) + e1;

lo2 = max(mean(err2,2) - e2,0);
hi2 = mean(err2,2) + e2;

lo3 = max(mean(err3,2) - e3,0);
hi3 = mean(err3,2) + e3;

hp3 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo3; hi3(end:-1:1); lo3(1)], col(3,:));
hl3 = plot(n_array,mean(err3,2), '-.', 'Color', col(3,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl3.LineWidth = 2;

hp2 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo2; hi2(end:-1:1); lo2(1)], col(2,:));
hl2 = plot(n_array,mean(err2,2), '--', 'Color', col(2,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl2.LineWidth = 2;

hp1 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo1; hi1(end:-1:1); lo1(1)], col(1,:));
hl1 = plot(n_array,mean(err1,2), 'Color', col(1,:));
% hl1 = errorbar(n_array,mean(err1,2), e1);
hl1.LineWidth = 2;



set(hp1, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
set(hp2, 'facecolor', [1.0 0.8 0.8], 'edgecolor', 'none');
set(hp3, 'facecolor', [0.95,0.95,0.6], 'edgecolor', 'none');

% alpha(0.8) 

% set(hl1, 'color', 'b');
% set(hl2, 'LineStyle', '--');

% xlabel("number of data points")
% ylabel('$ E \|D_{\phi_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',14)

switch breg
    case 'ItSa'
        title('Itakura-Saito distance')
    case 'Mahal'
        title('Mahalanobis distance')
    case 'GID'
        title('Generalized I-divergence')   
end
xlim([min_n,max_n])
% legend([hl1, hl2],["Bregman regression", "Mahalanobis regression"])
% legend




%% gid

subplot 132; hold on
load('results/regression/NBR_GID.mat');
err1 = err;
load('results/regression/MR_GID.mat');
err2 = err;
load('results/regression/MLP_GID.mat');
err3 = err(2:20,:);
clear err

breg = 'ItSa';
% breg = 'Mahal';
breg = 'GID';
num_run = 20;
min_n = 10;
max_n = 100;
n_array = (min_n:5:max_n);


n_array = n_array';
%% gid

% figure('Position', [0 0 350 300]);  hold on
col = get(gca,'colororder');

e1 = std(err1,0,2)/sqrt(num_run);
e2 = std(err2,0,2)/sqrt(num_run);
e3 = std(err3,0,2)/sqrt(num_run);


lo1 = max(mean(err1,2) - e1,0);
hi1 = mean(err1,2) + e1;

lo2 = max(0,mean(err2,2) - e2);
hi2 = mean(err2,2) + e2;

lo3 = max(mean(err3,2) - e3,0);
hi3 = mean(err3,2) + e3;

hp3 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo3; hi3(end:-1:1); lo3(1)], col(3,:));
hl3 = plot(n_array,mean(err3,2), '-.', 'Color', col(3,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl3.LineWidth = 2;

hp2 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo2; hi2(end:-1:1); lo2(1)], col(2,:));
hl2 = plot(n_array,mean(err2,2), '--', 'Color', col(2,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl2.LineWidth = 2;

hp1 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo1; hi1(end:-1:1); lo1(1)], col(1,:));
hl1 = plot(n_array,mean(err1,2), 'Color', col(1,:));
% hl1 = errorbar(n_array,mean(err1,2), e1);
hl1.LineWidth = 2;

set(hp1, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
set(hp2, 'facecolor', [1.0 0.8 0.8], 'edgecolor', 'none');
set(hp3, 'facecolor', [0.95,0.95,0.6], 'edgecolor', 'none');

% alpha(0.8) 

% set(hl1, 'color', 'b');
% set(hl2, 'LineStyle', '--');

% xlabel("number of data points")
% ylabel('$ E \|D_{\phi_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',14)

switch breg
    case 'ItSa'
        title('Itakura-Saito distance')
    case 'Mahal'
        title('Mahalanobis distance')
    case 'GID'
        title('Generalized I-divergence')   
end
xlim([min_n,max_n])
% legend([hl1, hl2],["Bregman regression", "Mahalanobis regression"])
% legend

%% mahal

subplot 133; hold on
load('results/regression/NBR_mahal.mat');
err1 = err;
load('results/regression/MR_mahal.mat');
err2 = err;
load('results/regression/MLP_mahal.mat');
err3 = err;

clear err

breg = 'ItSa';
breg = 'Mahal';
% breg = 'GID';
num_run = 20;
min_n = 5;
max_n = 100;
n_array = (min_n:5:max_n);


n_array = n_array';
%% mahal

% figure('Position', [0 0 350 300]);  hold on
 col = get(gca,'colororder');

e1 = std(err1,0,2)/sqrt(num_run);
e2 = std(err2,0,2)/sqrt(num_run);
e3 = std(err3,0,2)/sqrt(num_run);


lo1 = max(mean(err1,2) - e1,0);
hi1 = mean(err1,2) + e1;

lo2 = max(0,mean(err2,2) - e2);
hi2 = mean(err2,2) + e2;

lo3 = max(mean(err3,2) - e3,0);
hi3 = mean(err3,2) + e3;

hp3 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo3; hi3(end:-1:1); lo3(1)], col(3,:));
hl3 = plot(n_array,mean(err3,2), '-.', 'Color', col(3,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl3.LineWidth = 2;

hp2 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo2; hi2(end:-1:1); lo2(1)], col(2,:));
hl2 = plot(n_array,mean(err2,2), '--', 'Color', col(2,:));
% hl2 = errorbar(n_array,mean(err2,2), e2);
hl2.LineWidth = 2;

hp1 = patch([n_array; n_array(end:-1:1);n_array(1)], [lo1; hi1(end:-1:1); lo1(1)], col(1,:));
hl1 = plot(n_array,mean(err1,2), 'Color', col(1,:));
% hl1 = errorbar(n_array,mean(err1,2), e1);
hl1.LineWidth = 2;

set(hp1, 'facecolor', [0.8 0.8 1], 'edgecolor', 'none');
set(hp2, 'facecolor', [1.0 0.8 0.8], 'edgecolor', 'none');
set(hp3, 'facecolor', [0.95,0.95,0.6], 'edgecolor', 'none');

% alpha(0.8) 

% set(hl1, 'color', 'b');
% set(hl2, 'LineStyle', '--');

% xlabel("number of data points")
% ylabel('$ E \|D_{\phi_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',14)

switch breg
    case 'ItSa'
        title('Itakura-Saito distance')
    case 'Mahal'
        title('Mahalanobis distance')
    case 'GID'
        title('Generalized I-divergence')   
end
xlim([min_n,max_n])
legend([hl1, hl2, hl3],["Bregman regression", "Mahalanobis regression", "MLP regression"])
legend

%%

h ;
h.NextPlot = 'add';
a = axes; 

%// Set the title and get the handle to it
% ht = title('Test');
hx = xlabel('Number of training data');
hy = ylabel('$ E \|D_{\phi_n}(x_1,x_2)- D_{\phi_*}(x_1,x_2) \|_2^2$','Interpreter','latex', 'FontSize',14);

%// Turn the visibility of the axes off
a.Visible = 'off';

%// Turn the visibility of the title on
% ht.Visible = 'on';
hx.Visible = 'on';
hy.Visible = 'on';

% 
% vec_pos = get(get(gca, 'XLabel'), 'Position');
% 
% set(get(gca, 'XLabel'), 'Position', vec_pos + [0 0 0]);
