function [P] = softmaxUpd(X,Theta,b)
%-- X [nFea nSmp]
%-- Theta [nFea nClass]
%-- b [1 nClass]

%-- each entry pki = p(k|xi) = exp(theta_k'*xi +bk)/sum_j(theta_j'*xi +bj)
%-- P[nSmp nClass]: each row is summed to 1


P = exp(Theta'*X + repmat(b',1,size(X,2)))';         %-- by transpose, each row ith: theta_1'*xi  theta_2'*xi theta_3'*xi
P = bsxfun(@rdivide,P,sum(P,2)); %-- softmax for each row i: exp(theta_k'*xi)/sum_j(theta_j'*xi)

end
