
%% setup environment for the package

INSTALL

current_path=cd;
addpath(genpath(cd))
addpath(genpath('../LibData')) %-- add path to library files and datasets

%% Prepare simple data from school.mat for both (logistic) regression
% get only 2 first schools with 28 fea, 10 and 15 samples respectively
% 3 gnd: regression, 2-classes, 3-classes for binary and multiclass testing
% Too simple problem: like from original dataset, splitting them into 2
% partitions, and said it is a multi-task learning!

clear all;close all; clc;
load data/school.mat

a = X{1}; a(11:end,:) = []; b = X{2}; b(16:end,:) = []; 
data.X = {a b};
a = Y{1}; a(11:end,:) = []; b = Y{2}; b(16:end,:) = []; 
data.gndReg = {a b};


idx1 = find(a>=15); idx2 = find(a<15); a(idx1) = 0; a(idx2) = 1; 
idx1 = find(b>=25); idx2 = find(b<25); b(idx1) = 0; b(idx2) = 1; 
data.gnd2Class = {a b};


a = Y{1}; a(11:end,:) = []; b = Y{2}; b(16:end,:) = []; 
idx1 = find(a<10); idx2 = find(a<17 & a>=10); idx3 = find(a>=17); 
a(idx1) = 1; a(idx2) = 2; a(idx3) = 3; 

idx1 = find(b<25); idx2 = find(b<40 & b>=25); idx3 = find(b>=40); 
b(idx1) = 1; b(idx2) = 2; b(idx3) = 3; 
data.gnd3Class = {a b};

data.name = '[X gndReg] for regression, [X gnd2/3Class] for Classification';
save('data/2School.mat','data');


%% Test Least_TGL and Logistic_TGL (DONE for all Reg/2-3Class)
%% Test Logistic_TGL as well (DONE)
%- 4 & 5th fea are common in all reg/2-3Class cases
%- larger lambda, higher sparsity

clear all;close all; clc;
load data/2School.mat

X = data.X;
% Y = data.gndReg; 
Y = data.gnd3Class; 
lmda = [20:30:150];
lmda = lmda/10000; %- further for classification

d = size(X{1}, 2);  %- 28 features for each student

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.
WAll = [];     %-- tracking model's coeff

sparsity = zeros(length(lmda), 1);
log_lam  = log(lmda);

rho1= 0.1; rho2 = 1.1; 

for i = 1: length(lmda)
%     [W funcVal] = Least_L21(X, Y, lmda(i), opts);
%     [W funcVal] = Logistic_L21(X, Y, lmda(i), opts);
%     [W funcVal] = Least_TGL(X, Y, rho1, rho2, lmda(i),opts);
    [W,C,funcVal] = Logistic_TGL(X, Y, rho1, rho2, lmda(i),opts);
    
    opts.init = 1; % set the solution as the next initial point to get better efficiency. 
    WAll = [WAll W];
    sparsity(i) = nnz(sum(W,2 )==0)/d; %-- # zero-out features divided by d
end

WAll
sparsity'

% draw figure
h = figure;
plot(log_lam, sparsity);
xlabel('log(\lambda_1)'); ylabel('Percentage of All-0 Columns (row sparsity)')
title('Row Sparsity of Predictive Model when Changing Regularization Parameter');
set(gca,'FontSize',12); print('-dpdf', '-r100', 'LeastL21Exp');





%% DONE test on toyData, convert it to school format
%- perfectly with lmda = 0.00003; test other: lmda = [0.00001 0.00003 0.000035];

clear all;close all; clc;
load 'toyData'; dataToy = data;
load '2School'; dataSchool = data; clear data;

data.X{1} = dataToy.X(:,1:2)';
data.X{2} = dataToy.X(:,2:3)';
data.gnd2Class{1} = dataToy.gnd(1:2)';
data.gnd2Class{2} = dataToy.gnd(2:3)';
X = data.X; Y = data.gnd2Class; 

rho1= 0.1; rho2 = 1.1; lmda = 0.00003; opts ={};
[W,C,funcVal] = Logistic_TGL(X, Y, rho1, rho2, lmda,opts);
W


%%

