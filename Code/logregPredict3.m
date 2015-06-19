function [yhat, p] = logregPredict3(model, X)
%-- X [nSmp nFea] 
%-- for each xi as a row in X, find p(c|xi), so size of p is (n,k)
%-- yhat is the final class label for each xi, as the max_c(c|xi)

% Predict response for logistic regression
% p(i, c) = p(y=c | X(i,:), model)
% yhat(i) =  max_c p(i, c) 
%   For binary, this is {0,1} or {-1,1} or {1,2}, same as training
%   For multiclass, this is {1,..,C}, or same as training
%
% Any preprocessing done at training time (e.g., adding 1s,
% standardizing, adding kenrels, etc) is repeated here.

% This file is from pmtk3.googlecode.com

if ~strcmpi(model.modelType, 'logreg')
  error('can only call this funciton on models of type logreg')
end

if isfield(model, 'preproc')
    X = Xnorm(X',model.preproc); 
    X = X';
end

if size(model.w,2)==1  % binary 2-class model
    %-- 2 classes
    p = sigmoid(X*model.w);
    yhat = p > 0.5;  % now in [0 1]
    yhat = setSupport(yhat, model.ySupport, [0 1]); % restore initial support 
else 
    %-- p [nSmp nClass]
    p = softmaxPmtk(X*model.w); %-- ok, compute normalized exp(xi*wk) via softmax
    yhat = maxidx(p, [], 2); % yhat in 1:C
    C = size(p, 2); % now in 1:C
    yhat = setSupport(yhat, model.ySupport, 1:C); % restore initial support
end
end
