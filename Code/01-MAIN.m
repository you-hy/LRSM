%% add current path to run code from other works
addpath(genpath(cd))


%% --  LSRML1: toydata
%-- DONE! (identical)
% %{

%-- 1. Load data
close all; clear all; clc;
load data/syn3Gauss10D.mat

trainData.X = data.X;
trainData.gnd = data.gnd;
opt.verbose = 1;
opt.lmda1 = 0.1;
opt.lmda2 = 0;
opt.lambda = 0.1;
opt.eta = .01;

%-- 2. Train model
m2 = LRSML1(trainData,opt); %-- L1

% m2 = LRSM3(trainData,opt); m2.Theta = m2.w; %-- L2+SD+Armijio

%-- 3. Feature selection

nFeaSel = 30;
feaSel = zeros(size(data.X,1),1);
[idxSel idxDel sprw] = getTopFeaIdx(m2.w(2:end,:),nFeaSel); %
feaSel(idxSel) = 1; % store index of selected features

m2.w = [m2.w(1,:); sprw]; 
[yHat p] = logregPredict3(m2,data.XTest'); %[yHat yTest']
sum(yHat == data.gndTest')/length(data.gndTest);

%-- 4. Visualization

figure;
subplot(3,1,1); stem(abs(m2.w(2:end,1)),'Color','k');% xlim([.8 8.2])
subplot(3,1,2); stem(abs(m2.w(2:end,2)),'Color','k');% xlim([.8 8.2])
subplot(3,1,3); stem(abs(m2.w(2:end,3)),'Color','k');% xlim([.8 8.2])

%}





%% -- 07-ADNIshort with LRSM3
%-- DONE
%-- HY data:  max accu at lmda2=.1 (train/val/test)   1  .5292    0.4656
%          : nll reduces but g sth goes up and down (ovall reduces)
%-- YL data:  max accu at lmda2=.05 (train/val/test)   1  0.5415 0.6522
%          : nll reduces but g sth goes up and down (ovall reduces)


close all; clear all; clc;
load 07-ADNIshort
data = dataAll{2}; %-- Yilei data

% data = dataAll{1};%-- HY data with network
% data.W(:,:,1) = full(data.Wnc); 
% data.W(:,:,2) = full(data.Wmci);
% data.W(:,:,3) = full(data.Wad);

minCorr = 0;
[r c] = find(data.X <= minCorr);
for i=1:numel(r)
    data.X(r(i),c(i)) = 0;
end


% close all; clear all; clc;
% load data/syn3Gauss10D
% data = genCrossVal(data,3,1);


% close all; clear all; clc;
% load data/08-ADNIsyn
% data = genCrossVal(data,3,1);



[nFea nSmp] = size(data.X);
nFold = length(unique(data.testSet));

opt.verbose = 1;
opt.maxIter = 300;
opt.lmda1 = 0.1;
opt.lmda2 = 1.0;
opt.method = 'LRSM3';

lmda2Set = [0 .05 .1 .2 .5 1]; 
% lmda2Set = [0 .1]; 

nLmda2 = numel(lmda2Set);
nFeaSel = 2;


res.lmda2Set = lmda2Set;
res.acc = zeros(numel(lmda2Set),3); %-- 3: train/validate/test accuracy
res.feaSel = zeros(nFea,nLmda2);
res.feaModel = {};
res.nFeaSel = nFeaSel;


for iLmda = 1:nLmda2 %-- test impact of smoothness
    fprintf('Work at Lmda2: %3.2f ...\n',lmda2Set(iLmda));
    opt.lmda2 = lmda2Set(iLmda);
    
    %-- 1. Avg trainErr and valiErr
    m1 = accuracyALL(data,opt);
    res.acc(iLmda,1:2) = [mean(m1.accuCVTrain) mean(m1.accuCV)];
    
    %-- 2. using whole trainData for fea sele, not for accuracy!
    trainData.X=data.X;
    trainData.gnd = data.gnd;
    
    m2 = LRSM3(trainData,opt); %[m2.yHat trainData.gnd']
    
    [idxSel idxDel sprw] = getTopFeaIdx(m2.w(2:end,:),nFeaSel);
    res.feaSel(idxSel,iLmda) = 1; % store selFeatures
    res.feaModel{end+1} = sprw;
    
    m2.w(idxDel+1,:) = 0; % keep seleFea (+1 as 1st row is bias)
    
    %-- 3. Train/ValiErr for other classifiers
    %-- 4. testErr on testData
    [yHat p] = logregPredict3(m2,data.XTest'); %[yHat yTest']
    res.acc(iLmda,3) =  sum(yHat == data.gndTest')/length(data.gndTest);
end

% resHY = res;
resYL = res;
save('output/07-Lmda2Impact.mat','resHY','resYL'); disp('done');



figure;  plot(lmda2Set,res.acc(:,1)','r--+',lmda2Set,res.acc(:,2)','g--o',lmda2Set,res.acc(:,3)','k--*')
legend('training accuracy','CV accuracy','test accuracy');xlabel('\lambda_2'); ylabel('accuracy');






%% --  07-ADNIshort with LRSM3 vs logregFit
%-- DONE! (identical)
% %{

%-- 1. Load data
close all; clear all; clc;
% load 07-ADNIshort
% data = dataAll{2};

% minCorr = 0;
% [r c] = find(data.X <= minCorr);
% for i=1:numel(r)
%     data.X(r(i),c(i)) = 0;
% end 


% load data/syn3Gauss10D.mat

load data/08-ADNIsyn
data = dataAll{1};

trainData.X = data.X;
trainData.gnd = data.gnd;
opt.verbose = 1;
opt.lmda1 = 0.1;
opt.lmda2 = 0;
opt.lambda = 0.1;
opt.eta = .01;

%-- 2. Train model

% opt.regType='l2';opt.lambda = .1;
% m1 = LRSML2(trainData,opt); %-- old ver
% addpath(genpath('/home/dang/Dropbox/DQA/PP/Murphy')); %ensure path added
%  m2 = logregFit(trainData.X',trainData.gnd',opt); 

%  m2 = LRSML1(trainData,opt); %-- L1

m2 = LRSM3(trainData,opt); m2.Theta = m2.w; %-- L2+SD+Armijio

%-- 3. Feature selection

nFeaSel = 30;
feaSel = zeros(size(data.X,1),1);
[idxSel idxDel sprw] = getTopFeaIdx(m2.w(2:end,:),nFeaSel); %
feaSel(idxSel) = 1; % store index of selected features

m2.w = [m2.w(1,:); sprw]; 
[yHat p] = logregPredict3(m2,data.XTest'); %[yHat yTest']
sum(yHat == data.gndTest')/length(data.gndTest);

%-- 4. Visualization

figure;
subplot(3,1,1); stem(abs(m2.w(2:end,1)),'Color','k');% xlim([.8 8.2])
subplot(3,1,2); stem(abs(m2.w(2:end,2)),'Color','k');% xlim([.8 8.2])
subplot(3,1,3); stem(abs(m2.w(2:end,3)),'Color','k');% xlim([.8 8.2])

%}


%% --  3Gauss10D with LRSML2
%-- DONE! Converged
 %{

close all; clear all; clc;
load data/syn3Gauss10D.mat

trainData.X=data.X;
trainData.gnd = data.gnd;
opt.verbose = 1;
opt.lmda1 = 0.1;
opt.lmda2 = 0.5;
opt.eta = .1;

m2 = LRSML2(trainData,opt); %[m2.yHat trainData.gnd']
figure;
subplot(3,1,1); stem(abs(m2.w(:,1)),'Color','k'); xlim([.8 8.2])
subplot(3,1,2); stem(abs(m2.w(:,2)),'Color','k'); xlim([.8 8.2])
subplot(3,1,3); stem(abs(m2.w(:,3)),'Color','k'); xlim([.8 8.2])


nFeaSel = 2;
[idxSel idxDel sprTheta] = getTopFeaIdx(m2.Theta,nFeaSel);
m2.Theta(idxDel,:) = 0;

[yHat p] = logregPredict3(m2,data.XTest'); %[yHat yTest']
acc =  sum(yHat == data.gndTest')/length(data.gndTest)

figure;
subplot(1,2,1);PlotX(data.XTest,yHat,'','',''); grid on; title('prediction');
subplot(1,2,2);PlotX(data.XTest,data.gndTest,'','',''); grid on; title('groundtruth');



%}



%% --  07-ADNIshort with L2 LogReg (with L1, his algo is halt!)
%-- DONE achive 67.39% at lmda = .01 (lamda L2 did impact acc)
% %{

% addpath(genpath('/home/dang/Dropbox/DQA/PP/Murphy'));
close all; clear all; clc;
load 07-ADNIshort
data = dataAll{2};

minCorr = 0;
[r c] = find(data.X <= minCorr);
for i=1:numel(r)
    data.X(r(i),c(i)) = 0;
end

X = data.X'; y = data.gnd;
XTest = data.XTest'; yTest = data.gndTest;
lmdArry = [.005 .01 ];
res.accTrain = zeros(1,numel(lmdArry));
res.accTest = zeros(1,numel(lmdArry));

opt.regType='l2';
for i=1:numel(lmdArry)
    fprintf('\n Lambda2: %4.3f\n\n',lmdArry(i))
    opt.lambda = lmdArry(i);
    model = logregFit(X, y,opt);
    [outLabelTn p] = logregPredict(model, X);
    [outLabelTe p] = logregPredict(model, XTest);
    res.accTrain(i) = sum(outLabelTn==y')/length(y);
    res.accTest(i) = sum(outLabelTe==yTest')/length(yTest);
end

[res.accTrain' res.accTest']

%}

%% --  07-ADNIshort with various classifiers in AccuracyAll
%-- DONE
%-- res: {'SVM','DTREE','FOREST','LogRegL2'}; {66% 39% 51% 66%}
%-- If so, reporting how better it is compared to random method of 33%?
%-- Or p-value w.r.t. random classifiers?
%-- If guessing all to MCI: acc = 60/131 = 45.8%

% %{
close all; clear all; clc;
load 07-ADNIshort
data = dataAll{2};

% minCorr = 0; data = generateGraph(data,minCorr);

minCorr = 0;
[r c] = find(data.X <= minCorr);
for i=1:numel(r)
    data.X(r(i),c(i)) = 0;
end

datacmb.X = [data.X data.XTest];
datacmb.gnd = [data.gnd data.gndTest];
datacmb.testSet = [ones(1,length(data.gnd)) 2*ones(1,length(data.gndTest))];

clserSet = {'SVM','DTREE','FOREST','LogRegL2'}; %-- 66% 39% 51% 66%
clserSet = {'LRSM3'}; 
clserSet = {'LogRegL2'}; 

res = {};
for i=1:numel(clserSet)
 opt.method = clserSet{i};
 opt.idxFold = 2;
 [m] = accuracyALL(datacmb,opt)
 res = [res m];
end

%}


%% -- 07-ADNIshort with LRSML2
%--
close all; clear all; clc;
load 07-ADNIshort
data = dataAll{2};

data.W(:,:,1) = full(data.Wnc);
data.W(:,:,2) = full(data.Wmci);
data.W(:,:,3) = full(data.Wad);

[nFea nSmp] = size(data.X);

nFold = length(unique(data.testSet));

opt.verbose = 1;
opt.maxIter = 300;
opt.eta = 0.1;
opt.lmda1 = 0.1;
opt.lmda2 = 1.0;
opt.method = 'LRSML2';

lmda2Set = [.1 .5 1 2 5 10];
nLmda2 = numel(lmda2Set);

%-- train/validate/test accuracy
res.lda2Set = lmda2Set;
res.acc = zeros(numel(lmda2Set),3);
res.feaSel = zeros(nFea,nLmda2);
res.feaModel = {};

nFeaSel = 10;

for iLmda = 1:numel(lmda2Set)
    fprintf('Work at Lmda2: %3.2f ...\n',lmda2Set(iLmda));
    opt.lmda2 = lmda2Set(iLmda);
    
    %-- 1. Avg trainErr and valiErr
    m1 = accuracyALL(data,opt);
    res.acc(iLmda,1:2) = [mean(m1.accuCVTrain) mean(m1.accuCV)];
    
    %-- 2. select fea using whole trainData (single set)
    trainData.X=data.X;
    trainData.gnd = data.gnd;
    trainData.W = data.W;
    
    m2 = LRSML2(trainData,opt); %[m2.yHat trainData.gnd']
    [idxSel idxDel sprw] = getTopFeaIdx(m2.Theta,nFeaSel);
    res.feaSel(idxSel(2:end)-1,iLmda) = 1; % store selFeatures
    res.feaModel{end +1} = sprw;
    
    m2.Theta(idxDel,:) = 0;
    
    %-- 3. Train/ValiErr for other classifiers
    %-- 4. testErr on testData
    [yHat p] = logregPredict3(m2,XTest); %[yHat yTest']
    res.acc(iLmda,3) =  sum(yHat == yTest')/length(yTest);
end

save('output/07-ADNIres01.mat','res'); disp('done');

figure;  plot(lmda2Set,res.acc(:,1)','r--+',lmda2Set,res.acc(:,2)','g--o',lmda2Set,res.acc(:,3)','k--*')
legend('training accuracy','CV accuracy','test accuracy');



%% -- 3Gauss: gndFea in 1st 2 dims, last 8 ones are random
% DONE
% 2 separate datasets: training (for CV) and test
% Pipeline: 1. use CV-training data for opt param.
%           2. apply opt-param on whole training to get features
%           3. use sele fea for test data.
%           4. Plot error curves for training, validation and test

% %{

close all; clear all; clc;

% 1. Gen data: 3 10Dim classes in 3 corners, gndFea = {d1,d2}
mus     = [1 -1; -1 0; 0.5 1];   %--
nClass  = size(mus,1);
sigma   = 0.05;
N       = 200;
D       = 10;
pi      = ones(1,nClass)/nClass;
mus     = [mus zeros(nClass,D-size(mus,2))]; %- extending columns for means

randn('state',0);
X = zeros(N, D);
y = sampleDiscrete(pi, 1, N);
for c=1:nClass
    m.mu = mus(c, :);
    m.Sigma = sigma*eye(D);
    X(y==c, :) = gaussSample(m, sum(y==c));
end

testIdx = randi(N,1,100);
XTest = X(testIdx,:); X(testIdx,:)=[];
yTest = y(testIdx); y(testIdx) = [];


% 2. Model Evaluation

clc; data.X = X'; data.gnd = y;
data = genCrossVal(data,4,1);

[nFea nSmp] = size(data.X);

nFold = length(unique(data.testSet));

opt.verbose = 1;
opt.maxIter = 200;
opt.eta = 0.1;
opt.lmda1 = 0.01;
opt.lmda2 = 1.0;
opt.method = 'LRSML2';

lmda2Set = [.1 .5 .8 1 2];
nLmda2 = numel(lmda2Set);

%-- train/validate/test accuracy
res.lmda2Set = lmda2Set;
res.acc = zeros(numel(lmda2Set),3);
res.feaSel = zeros(nFea,nLmda2);
res.feaModel = {};


nFeaSel = 2;

for iLmda = 1:numel(lmda2Set)
    fprintf('Lambda: %3.2f \n',lmda2Set(iLmda));
    opt.lmda2 = lmda2Set(iLmda);
    
    %-- 1. Avg trainErr and valiErr
    m1 = accuracyALL(data,opt);
    res.acc(iLmda,1:2) = [mean(m1.accuCVTrain) mean(m1.accuCV)];

    %-- 2. select fea using whole trainData (single set), keep bias as well
    trainData.X=X'; trainData.gnd = y;
    m2 = LRSML2(trainData,opt); %[m2.yHat trainData.gnd']
    [idxSel idxDel sprw] = getTopFeaIdx(m2.Theta,nFeaSel);
    m2.Theta(idxDel,:) = 0;
    res.feaSel(idxSel(2:end)-1,iLmda) = 1; % excluding bias
    
    res.feaModel{end +1} = sprw;
    
    %-- 3. Train/ValiErr for other classifiers
    %-- 4. testErr on testData
    [yHat p] = logregPredict3(m2,XTest); %[yHat yTest']
    res.acc(iLmda,3) =  sum(yHat == yTest')/length(yTest);
end

save('output/vut.mat','res'); disp('done');

figure;  plot(lmda2Set,res.acc(:,1)','r--+',lmda2Set,res.acc(:,2)','g--o',lmda2Set,res.acc(:,3)','k--*')
legend('training accuracy','CV accuracy','test accuracy')



%Navigate to CGD folder and run (regression):
% [x,objective,ttime,mse] = CGD(y',X,X',10)
%}



%% -- 3Gauss: gndFea in 1st 2 dims, last 8 ones are random
% DONE
% Plot both training, validation and testing errors!
%{
close all; clear all; clc;

% 1. Gen data: 3 10Dim classes in 3 corners, gndFea = {d1,d2}
mus    = [1 -1; -1 0; 0.5 1];   %--
nClass = size(mus,1);
sigma = 0.05;
N = 200;
D   = 10;
pi = ones(1,nClass)/nClass;
mus = [mus zeros(nClass,D-size(mus,2))]; %- extending columns for means

randn('state',0);
X = zeros(N, D);
y = sampleDiscrete(pi, 1, N);
for c=1:nClass
    m.mu = mus(c, :);
    m.Sigma = sigma*eye(D);
    X(y==c, :) = gaussSample(m, sum(y==c));
end

testIdx = randi(N,1,100);
XTest = X(testIdx,:); X(testIdx,:)=[];
yTest = y(testIdx); y(testIdx) = [];


figure;
subplot(3,2,1);PlotX(X',y,'','',''); grid on;
subplot(3,2,2);PlotX(XTest',yTest,'','',''); grid on
subplot(3,2,3);PlotX(X(:,1:2)',y,'','',''); grid on;
subplot(3,2,4);PlotX(XTest(:,1:2)',yTest,'','',''); grid on
subplot(3,2,5);PlotX(X(:,2:3)',y,'','',''); grid on;
subplot(3,2,6);PlotX(XTest(:,2:3)',yTest,'','',''); grid on


% 2. Model training

clc; data.X = X'; data.gnd = y;
opt.verbose = 1; opt.maxIter = 2000;
opt.eta = 0.1;
opt.lmda1=0; opt.lmda2=1.0;
m = LRSML2(data,opt); m.Theta


figure;
subplot(3,1,1); stem(abs(m.Theta(:,1)),'Color','k'); xlim([.8 8.2])
subplot(3,1,2); stem(abs(m.Theta(:,2)),'Color','k'); xlim([.8 8.2])
subplot(3,1,3); stem(abs(m.Theta(:,3)),'Color','k'); xlim([.8 8.2])



% 3. Model test
[yHat, p] = logregPredict3(m, XTest);
figure;
subplot(1,2,1);PlotX(XTest',yHat,'','',''); grid on;
subplot(1,2,2);PlotX(XTest',yTest,'','',''); grid on

[yTest' yHat]


%Navigate to CGD folder (LRSM/Code/cgd_l1/cgd_l1) and run (regression):
[x,objective,ttime,mse] = CGD(y',X,X',10)

%}


%% Simple data with 8 nodes
%{
clear all;close all; clc;
load 'data/toyData'; whos
gObj = biograph(triu(data.W),data.feaName,'ShowArrows','off') % get(bg2.nodes,'ID')
gObj = view(gObj);
for i=1:size(data.X,1)
    a=corrcoef(data.X(i,:)',data.gnd'); a(1,2)
end


clc; opt.eta = 0.05; opt.verbose = 0; opt.maxIter = 2000;
opt.lambda1=0.01; opt.lambda2=0.8;
m = LRSML2(data,opt); [m.P data.gnd']

figure;
subplot(3,1,1); stem(abs(m.Theta(:,1)),'Color','k'); xlim([.8 8.2])
subplot(3,1,2); stem(abs(m.Theta(:,2)),'Color','k'); xlim([.8 8.2])
subplot(3,1,3); stem(abs(m.Theta(:,3)),'Color','k'); xlim([.8 8.2])

[m.Theta sum(m.Theta,2)]
%-- Q: multiple theta's, how to choose single set of features?
%-- Q: smoothness should exclude x0 (bias feature!) read Jyan
%}


%% Plot with double x-axis

figure;

Z = linspace(0,150)';           % Depth in meters
TC = -tanh((Z-30)/20)+23;       % Temperature in °C
dc2df = @(dc) (9/5)*dc + 32;    % °C->°F
mt2ft = @(mt) mt/0.3048;        % meters->feet

ax(1) = axes();
line(TC,Z,'parent',ax(1))
axis tight
xlabel('Temperature, °C')
ylabel('Depth, m')
ax(2) = axes('Position',get(ax(1),'position'),...
    'HitTest','off',...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'YDir','reverse',...
    'XLim',dc2df(get(ax(1),'XLim')),...
    'YLim',mt2ft(get(ax(1),'YLim')),...
    'Color','none');
xlabel(ax(2),'Temperature, °F')


%% -- 06-ADNI data

close all; clear all; clc;
load 06-ADNI
scaleFreeDisp(dataAll{1}.Wad)


X = dataAll{1}.X;
[row, col] = find(isnan(X));
X(row,col)


close all; clear all; clc;
load 07-ADNIshort

X = data.X;
X = data.XTest;
[r,c] = find(isnan(X));



%% -- 07-ADNIshort Preparation
% DONE
% From 06-ADNI dataAll{1}:
% - Divide 70% for training and 30% for test data
% - Further gen 5-CV on training data (for param tuning)
% - W of nc/mci/ad are in sparse format -> convert full and merge to 3D mat when loading

%{
%-- 1. data with Hongyuan pipeline

close all; clear all; clc;
load 06-ADNI
data.X = dataAll{1}.X;
[row, col] = find(isnan(data.X));
data.X(row,col) = 0;

data.gnd = dataAll{1}.gnd;

a = genCrossVal(data,10,1); %-- keep 70% for training 30% for test

testIdx = (a.testSet <=3);
trainIdx = ~testIdx;

data.XTest = double(data.X(:,testIdx));
data.gndTest = double(data.gnd(testIdx));

data.X = double(data.X(:,trainIdx));
data.gnd = double(data.gnd(trainIdx));
data.testSetPar = a.testSet;

data = genCrossVal(data,5,1); %-- 5-CV on training data

data.Wnc = sparse(dataAll{1}.Wnc);
data.Wmci = sparse(dataAll{1}.Wmci);
data.Wad = sparse(dataAll{1}.Wad);
data.name = 'ADNIshort 3 classes: 70%train (5-CV) + 30%test';

save('/home/dang/Dropbox/DQA/PP/LibData/Data/07-ADNIshort.mat','data');


% %-- 2. data with YiLei pipeline
%
% close all; clear all; clc;
% load 07-ADNIshort.mat
% data1 = data;
% clear data
%
% load 06-ADNI
% data.X = dataAll{3}.X;
% data.gnd = dataAll{3}.gnd;
% a = genCrossVal(data,10,1); %-- keep 70% for training 30% for test
%
% testIdx = (a.testSet <=3);
% trainIdx = ~testIdx;
%
% data.XTest = double(data.X(:,testIdx));
% data.gndTest = double(data.gnd(testIdx));
%
% data.X = double(data.X(:,trainIdx));
% data.gnd = double(data.gnd(trainIdx));
% data.testSetPar = a.testSet;
%
% data = genCrossVal(data,5,1); %-- 5-CV on training data
% data.name = 'ADNI Yilei 3 classes: 70%train (5-CV) + 30%test';
%
% clear dataAll
% dataAll{1} = data1;
% dataAll{2} = data;
%
%
% save('/home/dang/Dropbox/DQA/PP/LibData/Data/07-ADNIshort.mat','dataAll');


%}






%% --  07-ADNIshort with random fea for pvalue evaluation
%-- DONE: #Fea increases, TrainAcc increases, but TestAcc does not! 
%{
close all; clear all; clc;
load 07-ADNIshort
data = dataAll{2};
clserSet = {'SVM'}; 
opt.method = clserSet{1};
opt.idxFold = 2; %-- not CV, accTest on XTest
[nFea nSmp] = size(data.X);
datacmb.gnd = [data.gnd data.gndTest];
nTime   = 1000;
idxFea  = [1:nFea];

aTopFea = [10:5:50];
accTrain = zeros(nTime, numel(aTopFea)); % matrix of accu for each #topFea
accTest = zeros(nTime, numel(aTopFea)); % matrix of accu for each #topFea

for i=1:nTime
    for j=1:numel(aTopFea)
        fprintf('time %4d topFea %3d...\n',i,aTopFea(j));
        
        %-- randomly pickup aTopFea features
        rp = randperm(nFea);
        subNet = rp(1:aTopFea(j)); 

        datacmb.X = [data.X(subNet,:) data.XTest(subNet,:)];
        datacmb.testSet = [ones(1,length(data.gnd)) 2*ones(1,length(data.gndTest))];
        [m] = accuracyALL(datacmb,opt);
        accTrain(i,j) = m.accuCVTrain;
        accTest(i,j) = m.accuCV;
    end
    
end

save('output/07-randFea.mat','accTrain','accTest'); disp('done');

close all; clear all; clc
load output/07-randFea.mat

meanTrain = mean(accTrain);
stdTrain = std(accTrain);
meanTest = mean(accTest);
stdTest = std(accTest);
fprintf('\n   FeaSize     MeanTr      StdTr   MeanTe    StdTe\n')

[aTopFea' meanTrain' stdTrain' meanTest' stdTest']  

h=figure;set(h, 'Position', [230 250 1000 1000]);
for i=1:length(aTopFea)
    subplot(length(aTopFea),2,i); histfit(accTrain(:,i),50); 
    title(['FeaSet:',num2str(aTopFea(i)),...
        ' mean/std:', num2str(meanTrain(i),'%2.2f'),...
       ' (',num2str(stdTrain(i),'%2.2f'),'))']);
end


h=figure;set(h, 'Position', [230 250 1000 1000]);
for i=1:length(aTopFea)
    subplot(length(aTopFea),2,i); histfit(accTest(:,i),50); 
    title(['FeaSet:',num2str(aTopFea(i)),...
        ' mean/std:', num2str(meanTest(i),'%2.2f'),...
       ' (',num2str(stdTest(i),'%2.2f'),'))']);
end

%}


%% --  07-ADNIshort with CGD (regression + L1)
%-- DONE
%-- 524 fea selected, not evaluate regression yet
close all; clear all; clc;
load 07-ADNIshort

[x,objective,ttime,mse] = CGD(data.gnd',data.X',data.X,10);




%% --  08-ADNIsyn preparation
%-- DONE
%{
close all; clear all; clc;
load 08-ADNIsyn
data = dataAll{1}


a = genCrossVal(data,10,1); %-- keep 70% for training 30% for test

testIdx = (a.testSet <=3);
trainIdx = ~testIdx;

data.XTest = double(data.X(:,testIdx));
data.gndTest = double(data.gnd(testIdx));

data.X = double(data.X(:,trainIdx));
data.gnd = double(data.gnd(trainIdx));
data.testSetPar = a.testSet;

data = genCrossVal(data,5,1); %-- 5-CV on training data

data.Wnc = sparse(dataAll{1}.W);
data.Wmci = sparse(dataAll{1}.W);
data.Wad = sparse(dataAll{1}.W);

dataAll{1} = data;

save('/home/dang/Dropbox/DQA/PP/LibData/Data/08-ADNIsyn.mat','dataAll');

%}





%% --  08-ADNIsyn with LRSM3 
%-- DONE! (lack subnet visualization)
% %{

%-- 1. Load data
close all; clear all; clc;

% load data/syn3Gauss10D.mat
load data/08-ADNIsyn
data = dataAll{1};

trainData.X = data.X;
trainData.gnd = data.gnd;
opt.verbose = 1;
opt.lmda1 = 0.1;
opt.lmda2 = 0;
opt.lambda = 0.1;
opt.eta = .01;

%-- 2. Train model
%  m2 = LRSML1(trainData,opt); %-- L1
m2 = LRSM3(trainData,opt); m2.Theta = m2.w; %-- L2+SD+Armijio

%-- 3. Feature selection
nFeaSel = 30;
feaSel = zeros(size(data.X,1),1);
[idxSel idxDel sprw] = getTopFeaIdx(m2.w(2:end,:),nFeaSel); %
feaSel(idxSel) = 1; % store index of selected features

m2.w = [m2.w(1,:); sprw]; 
[yHat p] = logregPredict3(m2,data.XTest'); %[yHat data.gndTest']
sum(yHat == data.gndTest')/length(data.gndTest)

%-- 4. Visualization

figure;
subplot(3,1,1); stem(abs(m2.w(2:end,1)),'Color','k');% xlim([.8 8.2])
subplot(3,1,2); stem(abs(m2.w(2:end,2)),'Color','k');% xlim([.8 8.2])
subplot(3,1,3); stem(abs(m2.w(2:end,3)),'Color','k');% xlim([.8 8.2])


%}