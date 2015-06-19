function model = LogRegGradDesc(data, options)
%------------------------------------------------------------------------%
% Logistic Regression with Gradient Descent for multiClass case
% Minimize: NLL = - \sum_i \sum_k \log(p(k|xi)^{y_{ik}}) 
% Input:
%     + data: struct data type
%           .X[nFea nSmp]: dataset X 
%           .gnd[1 nSmp]:  class label
%     + options: structure
%           .eta: learning rate
%           .verbose: boolean, used to print out results
% Output:
%     + model: struct data type
%           .Theta[nFea, nClass]: each col ~ each model/class
%           .P[nSmp,nClass]: entry at row i col k: p(k|xi)
%                           max p(k|xi) ~ class label yi for xi
%                           check: [P data.gnd']
% Example: (c) 2015 Quang-Anh Dang - UCSB
%------------------------------------------------------------------------%

X = data.X;
y = data.gnd'; % to col vector format
[nFea,nSmp] = size(X);
% clear data;

X = Xnorm(X,1); %-- 0-mean 1-std for each X's fea 
X = [ones(1,nSmp); X]; %-- add x0 account for bias term/feature
nFea = nFea + 1;

if (~exist('options','var'))
    options = [];
end


eta = 0.01; % learning rate
if isfield(options,'eta'), 
    eta = options.eta; 
end


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


nClass = nunique(y);
yDummy = dummyEncoding(y, nClass); %-- 1-of-nClass coding scheme
Theta = rand(nFea,nClass);         %-- models to learn. theta_k in kth col 
P = softmaxUpd(X,Theta);           %-- initialize pki via softmax 

optTol = 1e-5*nClass;
iter = 0;


while iter < maxIter
    iter = iter + 1;
    Theta_old = Theta;
    for k=1:nClass
        diffk = P(:,k) - yDummy(:,k);
        dLk = sum(bsxfun(@times,X,diffk'),2); %-- X(pk - yk) = \sum_i(pki - yik)*x_i
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
model.P = P;
model.iter = iter;
model.name = 'MultiClass LogReg by Gradient Descent';

end

function [P] = softmaxUpd(X,Theta)
%-- each entry pki = p(k|xi) = exp(theta_k'*xi)/sum_j(theta_j'*xi)
%-- P[nSmp nClass]: each row is summed to 1

P = exp(Theta'*X)'; %-- row ith: theta_1'*xi  theta_2'*xi theta_3'*xi
P = bsxfun(@rdivide,P,sum(P,2)); %-- softmax for each row i: exp(theta_k'*xi)/sum_j(theta_j'*xi)

end