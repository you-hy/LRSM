% 1. Generate Synthetic Dataset
opt.nSmpEach = 50;
opt.nSmp = opt.nClass * opt.nSmpEach;

% Standard network for each class
Xstnd{1} = {[1,2,3],[1,8,4],[2,3,6],[2,8,2],[3,8,3],[3,4,3],[4,5,4],[5,6,2],[5,7,5],...
			[9,10,6],[9,11,5],[10,11,5],[10,12,5],[12,13,4],[12,17,3],[13,14,3],...
			[13,15,3],[14,15,2],[14,16,4],[15,16,5],[16,17,3],[17,18,5],[17,19,2],...
			[17,20,3],[18,19,3],[18,20,2]};
Xstnd{2} = {[1,2,2],[1,8,2],[2,3,5],[2,8,1],[3,4,2],[4,5,2],[5,7,4],[6,7,2],[9,10,4],...
			[9,11,3],[10,11,3],[10,12,3],[12,13,2],[13,14,2],[13,15,2],[14,16,3],...
			[15,16,3],[17,18,3],[18,19,2],[17,20,2]};
Xstnd{3} = {[1,2,1],[1,8,1],[2,3,3],[4,5,1],[5,7,2],[6,7,1],[9,10,3],[9,11,1],...
			[10,12,2],[12,13,2],[13,14,1],[14,16,2],[15,16,2],[16,20,2],[11,17,2],...
			[17,18,2],[17,20,1],[18,19,1]};
 opt.Xstnd = Xstnd;

% Generate ground-truth network (adj matrix) 			
addpath(genpath('../../LibData/Module'));
opt.nClass = 3;
opt.nFea = 20;
opt.gaussian_std = [1,1,1];
opt = gen_edge_sample(opt);

% Import rest nodes and scale-free network
net = importdata('../../R21 project/SNL/Matlab/Data/Generator/PA1000n20d50gs1000instGraph.csv');
opt.nRest = 1000;
Wrest = zeros(opt.nRest);
for i = 1:size(net,1)
	Wrest(net(i,1),net(i,2)) = 1;
end
Wrest = max(Wrest, Wrest');
Wrest = Wrest(1:opt.nRest ,1:opt.nRest);
Xrest = max(max(opt.X))*rand(opt.nRest,opt.nClass*opt.nSmpEach);

% Combine Wrest and Wstnd
data.X = [opt.X; Xrest];
W12 =  zeros(size(opt.W,1),size(Wrest,2));
W12(1,50) = 1;       W12(5,90) = 1;       W12(9,200)=1;
W12(17,250)=1;    W12(15,100)=1;     W12(20,300) = 1;
W12 = rand(size(W12)).*W12;
Wrest = rand(size(Wrest)).*Wrest;
data.W = [opt.W, W12; W12', Wrest];

data.gndFea = [ones(1,size(opt.W,1)), zeros(1,size(Wrest,2))];
data.gnd = [ones(1,opt.nSmpEach), 2*ones(1,opt.nSmpEach), 3*ones(1,opt.nSmpEach)];
data.name = 'synthetic data,  1000 random edges, 29 groundtruth edges, scale free';
dataAll{1} = data;

save('../../LibData/Data/08-ADNIsyn.mat','dataAll');