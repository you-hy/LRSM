%% 1. Generate Synthetic Dataset
opt.nClass = 3;
opt.nFea = 8;
opt.nSmpEach = 10;
opt.nSmp = opt.nClass * opt.nSmpEach;

% Standard network for each class
Xstnd{1} = {[1,2,3],[1,8,4],[2,3,6],[2,8,2],[3,8,3],[3,4,3],[4,5,4],[5,6,2],[5,7,5]};
Xstnd{2} = {[1,2,2],[1,8,2],[2,3,5],[2,8,1],[3,4,2],[4,5,2],[5,7,4],[6,7,2]};
Xstnd{3} = {[1,2,1],[1,8,1],[2,3,3],[4,5,1],[5,7,2],[6,7,1]};
opt.Xstnd = Xstnd;


Xstnd{1} = {[1,2,6],[1,3,5],[2,3,5],[2,4,5],[4,5,4],[4,9,3],[5,6,3],[5,7,3],[6,7,2],...
            [6,8,4],[7,8,5],[8,9,3],[9,10,5],[9,11,2],[9,12,3],[10,11,3],[10,12,2]};
Xstnd{2} = {[1,2,4],[1,3,3],[2,3,3],[2,4,3],[4,5,2],[5,6,2],[5,7,2],[6,8,3],[7,8,3],...
            [9,10,3],[10,11,2],[9,12,2]};
Xstnd{3} = {[1,2,3],[1,3,1],[2,4,2],[4,5,2],[5,6,1],[6,8,2],[7,8,2],[8,12,2],[3,9,2],...
            [9,10,2],[9,12,1],[10,11,1]};

% Generate samples
addpath(genpath('Module'));
opt.gaussian_std = [1,1,1];
opt = gen_edge_sample(opt);
opt.gnd = [ones(1,opt.nSmpEach), 2*ones(1,opt.nSmpEach), 3*ones(1,opt.nSmpEach)];
opt.edgeName = 1:10;

trainData.X	 = opt.X;
trainData.gnd = opt.gnd;
trainData.W = opt.W;
trainData.edgeName = opt.edgeName;

opt = gen_edge_sample(opt);
testData.X = opt.X;
testData.gnd = opt.gnd;	
testData.W = opt.W;

% 2. Model training and testing
addpath(genpath('../LRSM/Code'));
opt.nTop = 10;
options.eta = 0.1; 
options.verbose = 0; 
options.maxIter = 2000;
options.lmda2=0; 
options.lmda3=0;
model = model_LRSML2(trainData,options,opt);
test_LRSML2(trainData,testData,model);
[opt.edgeName',model.Theta(2:end,:)]

param.options = options;
param.trainData = trainData;
param.testData = testData;
param.opt = opt;
save('Module/ADNI-synthetic/toy.mat','param');

% 3. p-value test
n_pvalue = 100;
n_fea_pvalue = 5;
acc = zeros(1,n_pvalue);
for i = 1:n_pvalue
    index = randperm(10);
    index = index(1:n_fea_pvalue);
    
    svmtrainData.X = trainData.X(index,:);
    svmtrainData.gnd = trainData.gnd;
    % svmtrainData.gnd(svmtrainData.gnd==3) = 2;
    
    svmtestData.X = testData.X(index,:);
    svmtestData.gnd = svmtrainData.gnd;
    
    model = svmtrain(svmtrainData.gnd', svmtrainData.X', '-t 2');
    [~, accuracy, ~] = svmpredict(svmtestData.gnd', svmtestData.X', model);
    acc(i) = accuracy(1)/100;
end
mean(acc)


index = [6, 7, 8, 9, 10];
svmtrainData.X = trainData.X(index,:);
svmtrainData.gnd = trainData.gnd;
svmtrainData.gnd(svmtrainData.gnd==1) = 2;

svmtestData.X = testData.X(index,:);
svmtestData.gnd = svmtrainData.gnd;

model = svmtrain(svmtrainData.gnd', svmtrainData.X', '-t 0');
[~, accuracy, ~] = svmpredict(svmtestData.gnd', svmtestData.X', model);
acc = accuracy(1)/100;




