function [idxSel idxDel sprW] = getTopFeaIdx(W,nFeaSel,isAll)
%------------------------------------------------------------------------%
% Get top nFeaSel from Theta, 2 cases:
%   - C1: get each nFeaSel from each model and aggregate (isAll=0 or not
%   provided)
%   - C2: aggregate all models and get single nFeaSel (isAll=1)
% Input:
%     + W [nFea nClass]: each col represents for a model (eg. in LogReg)
%     + nFeaSel: Number of top features to be selected
%     + isAll: select nFeaSel over all models
% Output:
%     + idxSel: index of features including 1st for bias term
%     + idxDel: index of features to exclude/remove
% Example: (c) 2015 Quang-Anh Dang - UCSB
%------------------------------------------------------------------------%

if nargin==2, isAll = 0; end

% W(1,:) = []; % remove bias term b (done outside)
[nFea nCol] = size(W);

sprW = W;

for i=1:nCol % nFeaSel from each model/column
    [dump idx] = sort(abs(W(:,i)),'descend');
    sprW(idx(nFeaSel+1:end),i) = 0;
end

aggW = max(abs(sprW)')';

if isAll % exact nFeaSel from aggregated model
    [dump idx] = sort(aggW,'descend');
    idxSel = idx(1:nFeaSel)';
else    %-- aggregate FeaSel from all models (more than nFeaSel returned)
    idxSel = find(aggW>0)';    
end

idxDel = 1:nFea;
idxDel(idxSel) = [];


% if isAll % nFeaSel from ALL models
%     maxCoef = max(abs(Theta)')';
%     [dump idx] = sort(maxCoef,'descend');
%     
%     feaIncl = idx(1:nFeaSel)';
%     
%     feaExcl = idx(nFeaSel+1:end)';
% else % nFeaSel from each model and be combined
%     for i=1:nCol
%         [dump idx] = sort(abs(Theta(:,i)),'descend');
%         feaIncl = [feaIncl idx(1:nFeaSel)'];
%     end
%     feaIncl = unique (feaIncl);
%     feaExcl(feaIncl) =[];
% end

% idxSel = [1 1+ feaIncl]; %-- add bias as a must sel fea
% idxDel = 1+ feaExcl;

end