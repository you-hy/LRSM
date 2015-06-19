function model = LRSMV0(data, options)
%------------------------------------------------------------------------%
% Logistic Regression with Smooth Model via gradient descent
% Minimize: L = - \sum_i \sum_k \log(p(k|xi)^{y_{ik}}) 
%               + \lambda_1*\sum_k w_k^T*L*w_k
%               + \lambda_2*\sum_k^{K-1} \|w_k - w_{k+1}\|^2_2 
% 
% ==> For 3 classes, this learns 3 vectors w's and makes their coeff smooth
% but it may not work as each w classify 1 class from the other 2!
% Input:
%     + data: struct data type
%           .X[nFea nSmp]: dataset X 
%           .gnd[1 nSmp]: class label
%           .W[nFea nFea]: network topo
%     + options: structure
%           .lambda1:   L1 tradeoff for network 
%           .lambda2:   L2 tradeoff for smoothness
%           .eta: learning rate
%           .verbose: boolean, used to print out results
% Output:
%     + model: struct data type
%           .Theta[nFea, nClass]: each col for each model/class
%           .P[nSmp,nClass]: entry at row i col k: p(k|xi)
% Example: (c) 2015 Quang-Anh Dang - UCSB
%------------------------------------------------------------------------%


tic
%data = c2s(data);
X = data.X;
y = data.gnd'; % to col vector format
[nFea,nSmp] = size(X);

% clear data;

%-- 0-mean 1-std for each X's fea and add bias fea
X = Xnorm(X,1);

% %-- add bias term b = x0 
% X = [ones(1,nSmp); X];
% nFea = nFea + 1;


if (~exist('options','var'))
    options = [];
end

if isfield(data,'W'),
    Wc = max(data.W,data.W');
    
    %-- add & make biased-fea connect to NO other nodes
    %-- Q: should add 1 or 0?!
%     Wc = [zeros(1,nFea-1); Wc];
%     Wc = [zeros(nFea,1) Wc];
    
%     Wc = [ones(1,nFea-1); Wc];
%     Wc = [ones(nFea,1) Wc];
    
    
    Wc(1,1) = 0;
else
    Wc = eye(nFea);
end




lambda1 = 0.1; % tradeoff network
if isfield(options,'lambda1'),
    lambda1 = options.lambda1;
end

lambda2 = 0.1; % tradeoff model smoothness
if isfield(options,'lambda2'),
    lambda2 = options.lambda2;
end

eta = 0.01; % learning rate
if isfield(options,'eta'), 
    eta = options.eta; 
end


%-- construct graph-constrainted Laplacian C=Lc=Dc-Wc
Dc = full(sum(Wc,2));
C = -Wc; 
for j = 1 : size(C,1)
	C(j,j) = Dc(j) + C(j,j);
end

C
% C = [ones(size(C,1),1) C]; %-- add vector 1 accounting for bias feature x0

clear Dc Wc

maxIter = 1000;
if isfield(options,'maxIter'), 
    maxIter = options.maxIter; 
end


verbose = 0;
if isfield(options,'verbose'),
    verbose = options.verbose; 
end

if verbose       %Start the log
    fprintf('%10s %15s %15s\n','iter','sum(|Theta|)','sum(|Theta-Theta_old)');
end



% X[nFea nSmp]; yDummy[nSmp nClass]; P[nSmp nClass]; Theta[nFea nClass]


nClass = numel(unique(y));
yDummy = dummyEncoding(y, nClass); %-- 1-of-nClass coding scheme
Theta = rand(nFea,nClass);         %-- models to learn. theta_k in kth col 
P = softmaxUpd(X,Theta);           %-- initialize pki via softmax 

maxIter = 1000;
optTol = 1e-5*nClass;
iter = 0;


while iter < maxIter
    iter = iter + 1;
    Theta_old = Theta;
    
    Ctheta = C*Theta; %-- C'*[theta1 theta2 theta3] note symm C
    %ThetaDiff = bsxfun(@minus,Theta(:,1:end-1), Theta(:,2:end)); %-- theta_i - theta_{i+1}
    
    
    for k=1:nClass
        diffk = P(:,k) - yDummy(:,k);
        dLk = sum(bsxfun(@times,X,diffk'),2); %-- X(pk - yk) = \sum_i(pki - yik)*x_i
        dLk = dLk + lambda1*Ctheta(:,k);
        if (k==1)
            dLk = dLk + lambda2*(Theta(:,1) - Theta(:,2));
        elseif(k==nClass)
            dLk = dLk + lambda2*(Theta(:,end) - Theta(:,end-1));
        else
            dLk = dLk + lambda2*(2*Theta(:,k) - Theta(:,k+1) - Theta(:,k-1));
        end
        
        Theta(:,k) = Theta_old(:,k) - eta*dLk;
    end
    
    % update P_{ki} via softmax function using new Theta
    P = softmaxUpd(X,Theta);

    if verbose    
        fprintf('%10d  %15.6f %15.6f\n',iter,sum(sum(abs(Theta))),sum(sum(abs(Theta-Theta_old))));
    end
    
    
    % Check termination
    if sum(sum(abs(Theta-Theta_old))) < optTol
        break;
    end
    
end

model.Theta = Theta;
model.iter = iter;
model.P = P;
end

function [P] = softmaxUpd(X,Theta)
%-- each entry pki = p(k|xi) = exp(theta_k'*xi)/sum_j(theta_j'*xi)
%-- P[nSmp nClass]: each row is summed to 1

P = exp(Theta'*X)'; %-- row ith: theta_1'*xi  theta_2'*xi theta_3'*xi
P = bsxfun(@rdivide,P,sum(P,2)); %-- softmax for each row i: exp(theta_k'*xi)/sum_j(theta_j'*xi)

end