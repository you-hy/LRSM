function model = LRSML2(data, opt)
%------------------------------------------------------------------------%
% Logistic Regression with Smooth Model via gradient descent (L2 version)
% Minimize: L = - \sum_i \sum_k \log(p(k|xi)^{y_{ik}}) 
%               + \lmda_1*\sum_k w_k^T*C*w_k
%               + \lmda_2*\sum_k^{K-1} \|w_k - w_{k+1}\|^2_2 
%               + \\beta*\sum_k^{K-1} \|w_k\| (not implemented) 
% CONTINUE FROM HERE: 
%       - separate bias b from each theta as we need theta smooth, not bias
%       - assume 2 models: 
%           + by smoothness constraints, thetas are
%       similar/identical but it is bias b that makes hyperplane
%       different(actually parallel)
% 
% Input:
%     + data: struct data type
%           .X[nFea nSmp]: dataset X 
%           .gnd[1 nSmp]: class label
%           .W[nFea nFea nClass]: network topo for nClass
%     + opt: structure
%           .lmda1: L2 tradeoff for network topo C
%           .lmda1: L2 tradeoff for smoothness
%           .beta:  L1 tradeoff for fea selection
%           .eta:   learning rate
%           .verbose: boolean, used to print out results
% Output:
%     + model: struct data type
%           .Theta[1+nFea, nClass]: (1st for bias) each col for each model/class
%           .P[nSmp,nClass]: entry at row i col k: p(k|xi)
%           .preproc: Xmu/Xstnd of data.X (used to normalize Xtest later)
% Example: (c) 2015 Quang-Anh Dang - UCSB
%------------------------------------------------------------------------%


tic
%data = c2s(data);
X = data.X;
y = data.gnd'; % [nSmp 1] 
nClass = length(unique(y));
[nFea,nSmp] = size(X);

% clear data;

%-- 0-mean 1-std for each X's fea and add bias fea
preproc.standardizeX = 1;
[X,preproc] = Xnorm(X,preproc);
preproc.addOnes = 1; 


% %-- add bias term b = x0 
% X = [ones(1,nSmp); X];
% nFea = nFea + 1;


if (~exist('opt','var'))
    opt = [];
end

%-- build each Lapl.mat C for each class
C = zeros(nFea,nFea,nClass);
if isfield(data,'W'),
    if size(data.W,3)~=nClass
        error('Mismatch #networks and nClass!');
%     else
%         for i=1:nClass % ensure symm (undirected network)
%             data.W(:,:,i) = max(data.W(:,:,i),data.W(:,:,i)');
%         end
    end
    
    for i=1:nClass
        Dtmp = full(sum(data.W(:,:,i),2));
        Ctmp = -data.W(:,:,i);
        for j = 1 : size(Ctmp,1)
            Ctmp(j,j) = Dtmp(j) + Ctmp(j,j);
        end
        C(:,:,i) = Ctmp;
        % C = [ones(size(C,1),1) C]; %-- add vector 1 accounting for bias feature x0
    end
    clear Dtmp Ctmp
    
    
    %-- add & make biased-fea connect to NO other nodes
    %-- Q: should add 1 or 0?!
%     Wc = [zeros(1,nFea-1); Wc];
%     Wc = [zeros(nFea,1) Wc];
    
%     Wc = [ones(1,nFea-1); Wc];
%     Wc = [ones(nFea,1) Wc];
    
else
%     data.W = repmat(eye(nFea),1,1,3); %- not run on old matlab
    for i=1:nClass, C(:,:,i) = eye(nFea);   end
end

%-- construct graph-constrainted Laplacian C=Lc=Dc-Wc
%-- check C: a = C(:,:,1)*ones(nFea,1);

lmda1 = 0.1; % tradeoff network
if isfield(opt,'lmda1'),
    lmda1 = opt.lmda1;
end

lmda2 = 0.1; % tradeoff model smoothness
if isfield(opt,'lmda2'),
    lmda2 = opt.lmda2;
end

eta = 0.01; % learning rate
if isfield(opt,'eta'), 
    eta = opt.eta; 
end

maxIter = 1000;
if isfield(opt,'maxIter'), 
    maxIter = opt.maxIter; 
end

verbose = 0;
if isfield(opt,'verbose'),
    verbose = opt.verbose; 
end

if verbose       %Start the log
%     fprintf('%5s %15s %15s %5s \n','iter','sum(|Theta|)','sum(|Theta-Theta_old|)','NLL');
    fprintf('%5s %12s %12s \n','iter','NLL','DifNLL');    
end

% X[nFea nSmp]; yDummy[nSmp nClass]; P[nSmp nClass]; Theta[nFea nClass]

nClass = length(unique(y));
yDummy = dummyEncoding(y, nClass);  %-- 1-of-nClass coding scheme
Theta = zeros(nFea,nClass);         %-- models to learn. theta_k in kth col 
b = zeros(1,nClass);                %-- bias terms

optTol = 1e-5*nClass;
iter = 0;

[P,NLL] = softmaxUpdFull(X,yDummy,Theta,b,C,lmda1,lmda2);

% P = softmaxUpd(X,Theta,b);          %-- initialize pki via softmax 
% NLL = ComputeNLL(P,yDummy,Theta,C,lmda1,lmda2);


%Ctheta = zeros(nFea,nClass);

while iter < maxIter
    iter = iter + 1;
    prevTheta = Theta;
    prevb = b;
    prevNLL = NLL;
    sumTheta = sum(sum(abs(prevTheta))); % to check convergence
    
%     Ctheta = C*Theta; %-- C'*[theta1 theta2 theta3] note symm C
    
    for k=1:nClass
        % 1. Compute 1st derivative
        diffk = P(:,k) - yDummy(:,k);           % (pk - yk), colvec [nSmp 1] 
        dLk = sum(bsxfun(@times,X,diffk'),2);   % X(pk  - yk) = \sum_i(pki - yik)*xi: colvec[nFea 1] 
        dLk = dLk + lmda1*C(:,:,k)*Theta(:,k);  % add deriv of network term
        
        %-- add derv of tempo smooth term
        if (k==1)
            dLk = dLk + lmda2*(Theta(:,1) - Theta(:,2));
        elseif(k==nClass)
            dLk = dLk + lmda2*(Theta(:,end) - Theta(:,end-1));
        else
            dLk = dLk + lmda2*(2*Theta(:,k) - Theta(:,k+1) - Theta(:,k-1));
        end
        
        % 2. update param with 1st derivative
        Theta(:,k) = prevTheta(:,k) - eta*dLk;
        b(k) = prevb(k) - eta*sum(diffk);  %-- bN = bO - eta*sum_i(y(i,k) - p(k|xi))
    end
    
    %4. update P(k|i) via softmax using new Theta
%     P = softmaxUpd(X,Theta,b);
%     NLL = ComputeNLL(P,yDummy,Theta,C,lmda1,lmda2);
    [P,NLL] = softmaxUpdFull(X,yDummy,Theta,b,C,lmda1,lmda2);

    if verbose    
%         fprintf('%5d  %15.6f %15.6f  %10.5f\n',...
%             iter,sum(sum(abs(Theta))),...
%             sum(sum(abs(Theta-Theta_old))),nll);
        fprintf('%5d  %12.4f %12.4f\n',...
            iter,NLL, prevNLL - NLL);
    end
    
    % Check convergence
    if (iter>1) & (sum(sum(abs(Theta))) > sumTheta + 5.1*sumTheta) 
        % newTheta > 1,1*oldTheta -> consider diverged
        fprintf('Theta coeffs are diverged \n');
        break;
    end
    
    % Check termination
    if sum(sum(abs(Theta-prevTheta))) < optTol
        break;
    end
    
end

model.Theta = [b; Theta];
model.w = [b; Theta]; %-- for class border plotting
model.iter = iter;
model.P = P;
model.nclasses = nClass;
model.ySupport = (1:nClass)';
model.modelType = 'logreg';
model.preproc = preproc;
model.yHat = maxidx(model.P, [], 2); 
model.accTrain = sum(model.yHat'==data.gnd)/length(data.gnd);

end

function [P] = softmaxUpd(X,Theta,b)
%-- X [nFea nSmp] / Theta [nFea nClass] / b [1 nClass]
%-- P[nSmp nClass]: each row is summed to 1

[nFea nSmp] = size(X);
eta = Theta'*X + repmat(b(:),1,nSmp); % eta(i,k) = Theta(k)'*xi
eta = eta - repmat(max(eta),size(eta,1),1); %-- to avoid overflow of too large exp(.)


P = exp(eta)';         %-- by transpose, each row ith: theta_1'*xi  theta_2'*xi theta_3'*xi
P = bsxfun(@rdivide,P,sum(P,2)); %-- softmax Pki = exp(theta_k'*xi)/sum_j(exp(theta_j'*xi))


% W = Theta;
% eta = X'*W;
% Z = sum(exp(eta), 2);
% nclasses = size(eta,2);
% L = eta - repmat(log(Z), 1, nclasses);

%-- avoid overflow (like exp(1500) very very large) by substracting for the largest number:
% x = [305 300 150]
% exp(x)/sum(exp(x))
% x = x - max(x)
% exp(x)/sum(exp(x))

end

function [nll] = ComputeNLL(P,yDummy,Theta,C,lmda1,lmda2)
nClass = size(C,3);

%-- DQA!! issue here log(0) = -inf!!!
nll = - sum(log(sum(P.*yDummy,2))); % - sum_i log(P(k|i))
for k=1:nClass
    %-- add network term
    nll = nll + lmda1*Theta(:,k)'*C(:,:,k)*Theta(:,k);
    %-- add smoothness term
    if (k==1)
        dTheta = Theta(:,1) - Theta(:,2);
        L2Theta = dTheta.^2;
    elseif(k==nClass)
        dTheta = Theta(:,end) - Theta(:,end-1);
        L2Theta = dTheta.^2;
    else
        dTheta1 = Theta(:,k) - Theta(:,k-1);
        dTheta2 = Theta(:,k) - Theta(:,k+1);
        L2Theta = dTheta1.^2 + dTheta2.^2;
    end
    nll = nll + lmda2*sum(L2Theta);
end
end


function [P,nll] = softmaxUpdFull(X,yDummy,Theta,b,C,lmda1,lmda2)
%-- X [nFea nSmp] / Theta [nFea nClass] / b [1 nClass]
%-- P[nSmp nClass]: each row is summed to 1

[nFea nSmp] = size(X);
[nFea nClass] = size(Theta);

eta = Theta'*X + repmat(b(:),1,nSmp); % eta(i,k) = Theta(k)'*xi
eta = eta - repmat(max(eta),size(eta,1),1); %-- to avoid overflow of too large exp(.)
eta = eta'; %-- by transpose, each row ith: theta_1'*xi  theta_2'*xi theta_3'*xi
Z = sum(exp(eta),2);

lli = eta - repmat(log(Z),1,nClass); %-- (i,k): xi'*thetak - log sumj(exp(xi'thetaj))
P = exp(lli); %-- softmax Pki = exp[xi'*thetak -log sumj(exp(xi'thetaj))]

ll = sum(lli.*yDummy,2);
nll = -sum(ll);


for k=1:nClass
    %-- add network term
    nll = nll + lmda1*Theta(:,k)'*C(:,:,k)*Theta(:,k);
    %-- add smoothness term
    if (k==1)
        dTheta = Theta(:,1) - Theta(:,2);
        L2Theta = dTheta.^2;
    elseif(k==nClass)
        dTheta = Theta(:,end) - Theta(:,end-1);
        L2Theta = dTheta.^2;
    else
        dTheta1 = Theta(:,k) - Theta(:,k-1);
        dTheta2 = Theta(:,k) - Theta(:,k+1);
        L2Theta = dTheta1.^2 + dTheta2.^2;
    end
    nll = nll + lmda2*sum(L2Theta);
end


end
