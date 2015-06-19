function [ nzProb, maxProb ] = StabSelect( X, Y, opts )
% Function StabSelect: 
%   Longtitudinal Stability Selection 
%
%% Input parameters:
%   X: n * d data matrix;
%   Y: d * k target matrix;
%   opts.reg_factor:      
%   opts.iteration_times: 
%   opts.lambda_range:
%
%% Output parameters
%   nzProb       Selection probility for each regularization factor. 
%   maxProb      Selection score $\mathcal{S}$
%
%% Copyright (C) 2009-2011 Jiayu Zhou, Jun Liu, and Jieping Ye
%
%% Related functions:
%
% TGL
%
%%

%% parameter settings
reg_factor_1 = 0; 
reg_factor_2 = 0;
iteration_times = 1000;
d = size(X, 2);
k = size(Y, 2);
lambda_range = [  0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 ];
q = 2;

if isfield(opts, 'lambda_range')
    lambda_range = opts.lambda_range;
end

if isfield(opts, 'iteration_times')
    iteration_times = opts.iteration_times;
end

if isfield(opts, 'q')
    q = opts.q;
end

if q >= 10e6
    fprintf('Stability selection using (1, infinity)-norm \n');
else
    fprintf('Stability selection using (1, %u)-norm \n', q);
end

%% Lambda Loop. 
% probability table for all features over all lambda settings.
nzProb = zeros(d, length(lambda_range));

for i = 1:length(lambda_range)
    %% Probability Loop.
    % frequency table for all features over all iterations under current
    % lambda setting.
    nzFreq = zeros(d, iteration_times);
    for j = 1: iteration_times
        %prepare data.
        %bootstrap.
        bs_ind = rand(size(X,1),1)>=0.5;
        bsX = X(bs_ind, :);
        bsY = Y(bs_ind,:);
        
        aX= [];
        aY= [];
        ind = zeros(k, 1);
        for t = 1:k
            missing = bsY(:,t)<0;
            Xt = bsX(~missing, :);
            Xt = [zscore(Xt) ones(size(Xt, 1), 1)];
            Yt = bsY(~missing, t);
            aX = cat(1, aX, Xt);
            aY = cat(1, aY, Yt);
            ind(t+1) = ind(t) + size(Xt, 1);
        end
        
        opts.init=2;
        opts.tFlag=5;       % run .maxIter iterations
        opts.maxIter=100;   % maximum number of iterations
        opts.nFlag=0;       % without normalization
        opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
        opts.q=q;           % set the value for q
        opts.ind=ind;% set the group indices. 
        opts.z1=reg_factor_1;
		opts.z2=reg_factor_2;
        
        fprintf('Lambda = %.4f \t Iteration = %u \n',lambda_range(i),j);
        W = TGL(aX, aY, lambda_range(i), opts);
        
        % Homogenous contribution.
        % only select those whose signs of weights do not change over time.
        select = abs(max(sign(W),[],2)-min(sign(W),[],2))~=2 & sum(W~=0,2)>0;
        select = select(1:end-1); % remove bias.
        nzFreq(:, j) = select;
        
    end
    %calculate probability for each parameter in this lambda
    nzProb(:, i) = sum(nzFreq, 2)/iteration_times;
end
maxProb = max(nzProb,[], 2);
end

