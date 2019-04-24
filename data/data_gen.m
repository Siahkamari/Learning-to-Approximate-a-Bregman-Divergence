close all
clear, clc

%%
m1 = 500;
m2 = 500;
m3 = 500;
n = 4;
%%

p1 = [0.5, 0.5, 0.0];
p2 = [0.5, 0.0, 0.5];
p3 = [0.0, 0.5, 0.5];

X = [mnrnd(n,p1,m1);mnrnd(n,p2,m2);mnrnd(n,p3,m3)];
y = [ones(m1,1);2*ones(m2,1);3*ones(m3,1)];

inds = {};



Xp = X + 0.1*randn(m1+m2+m3,3);
for i=1:3
    scatter(Xp(inds{i},1),Xp(inds{i},2)); hold on 
end 
for i=1:3
    scatter(X(inds{i},1),X(inds{i},2)); hold on 
end 


X = X(:,1:2);

for r=1:4
   X = [X;X]
   y = [y;y]
end


%% circular
m1 = 200;
m2 = 200;
m3 = 200;

r = 1;
theta1 = 2*pi/3*rand(m1,1);
theta2 = 2*pi/3*rand(m2,1) + 2*pi/3;
theta3 = 2*pi/3*rand(m3,1) + 4*pi/3;
theta = [theta1;theta2;theta3];

X = r*[cos(theta), sin(theta)];
y = [ones(m1,1);2*ones(m2,1);3*ones(m3,1)];

for i=1:3
    inds = find(y == i);
    scatter(X(inds,1),X(inds,2)); hold on 
end 