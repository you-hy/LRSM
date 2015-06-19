function [model, X, lambdaVec, res] = LRSMVer16(data, opt)
%-- for l2: perform gradDesc with f'=g as initialized in penalizedL2 (no Hessian)
%--     for subsequent g, used to find next param w, are updated gradually via L-BFGS + lineSearch
%-- for l1: use 2-Metric Projection method w/ non-negative variables (see
%       Eq(12) in Yuan survey, or Mark Schmidt). Converting L1 to smooth constraints to optimize.


%%

%-- SD + Armijior linesearch

%-- 0-mean 1-std for each X's fea and add bias fea
[nFea nSmp] = size(data.X);
preproc.standardizeX = 1;
preproc.addOnes = 1; 
[X,preproc] = Xnorm(data.X,preproc);
preproc
size(X)



X = X'; y = data.gnd(:); 

% %-- set add bias and Xnorm
% preproc.standardizeX = 1;
% preproc.addOnes= 1; %-- add bias


nClass = nunique(y);
yDummy = dummyEncoding(y, nClass); %-- yi = (0,1,0) if xi is in 2nd class


% [preproc, X] = preprocessorApplyToTrain(preproc, X);


%--Now X is norm (m.0/std.1) + 1st col for bias with const 1/ std.0

nFea = size(X,2); % 1+nFea: as adding bias

lambdaVec = opt.lambda*ones(nFea, nClass); %-- also include bias term!
if preproc.addOnes
    lambdaVec(1, :) = 0; % don't penalize bias term
end

winit  = zeros(nFea, nClass);

%-- param for line search
verboseI = 0;
debug =  0;
doPlot = 0;
maxFunEvals = 2000;
maxIter =   200;
optTol =   1.0000e-05;
progTol =   1.0000e-09;
c1 =   1.0000e-04;
LS_interp =     2;
LS_multi =     0;

% Initialize
w = winit(:);

% If necessary, form numerical differentiation functions
funEvalMultiplier = 1;

data.X = X;
data.yDummy = yDummy;
data.lambda = lambdaVec(:);

[f,g] = penalizedL2(w,data);%-- nll, 1st derv from initial param
 
funEvals = 1;

% Output Log
if verboseI
    fprintf('%10s %10s %15s %15s %15s\n','Iteration','FunEvals','Step Length','Function Val','Opt Cond');
end

% Check optimality of initial point
if max(abs(g)) <= optTol
    fprintf('Optimality Condition below optTol \n');
    return;
end


for i = 1:maxIter
    wOld = w;
    %-- 1. DESCENT DIRECTION
    d = -g; 
    if ~isLegal(d),error('Step direction is illegal');  end
    
    %-- 2. COMPUTE STEP LENGTH 
    gtd = g'*d; 
    
    % Check that progress can be made along direction
    if gtd > -progTol,disp('Directional Derivative below progTol');break;end
    
    if i == 1 % Select Initial Guess
        t = min(1,1/sum(abs(g)));
    else
        t = min(1,2*(f-f_old)/(gtd));% See minFun for other t's
        if t <= 0,t = 1;end
    end
    
    
    fr = f; % Compute reference fr if using non-monotone objective
    
    %-- Line Search
    f_old = f;
    [t,w,f,g,LSfunEvals] = ArmijoBT(w,t,d,f,fr,g,gtd,...
        c1,LS_interp,LS_multi,progTol,debug,doPlot,1,data);
    funEvals = funEvals + LSfunEvals;
    
    fprintf('iter: %4d   nll: %10.5f   wDiff: %8.4f\n',i,f,sum((w-wOld).^2));
   
    
    if max(abs(g)) <= optTol, disp('Optimality Condition below optTol');break;end % Check Optimality Condition
    if max(abs(t*d)) <= progTol, disp('Step Size below progTol');break;end
    if abs(f-f_old) < progTol, disp('Function Value changing by less than progTol');break;end
    if funEvals*funEvalMultiplier >= maxFunEvals
        disp('Reached Maximum Number of Function Evaluations');break;
    end
    if i == maxIter,disp('Reached Maximum Number of Iterations');end
    
end


w = reshape(w, [nFea nClass]);
model.nclasses  = nClass;
model.w = w;
model.preproc = preproc;
model.f = f;





end


%-------------------------------------------------------------------------%

function [t,x_new,f_new,g_new,funEvals,H] = ArmijoBT(...
    x,t,d,f,fr,g,gtd,c1,LS_interp,LS_multi,progTol,debug,doPlot,saveHessianComp,data)

%-- KEY here: update new param by t*d, instead of learning rate!
%     [f_new,g_new] = funObj(x+t*d);

[f_new,g_new] = penalizedL2(x+t*d,data);

funEvals = 1;

while f_new > fr + c1*t*gtd || ~isLegal(f_new)
    temp = t;
    
    if LS_interp == 0 || ~isLegal(f_new)
        % Ignore value of new point
        if debug
            fprintf('Fixed BT\n');
        end
        t = 0.5*t;
    elseif LS_interp == 1 || ~isLegal(g_new)
        % Use function value at new point, but not its derivative
        if funEvals < 2 || LS_multi == 0 || ~isLegal(f_prev)
            % Backtracking w/ quadratic interpolation based on two points
            if debug
                fprintf('Quad BT\n');
            end
            t = polyinterp([0 f gtd; t f_new sqrt(-1)],doPlot,0,t);
        else
            % Backtracking w/ cubic interpolation based on three points
            if debug
                fprintf('Cubic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new sqrt(-1); t_prev f_prev sqrt(-1)],doPlot,0,t);
        end
    else
        % Use function value and derivative at new point
        
        if funEvals < 2 || LS_multi == 0 || ~isLegal(f_prev)
            % Backtracking w/ cubic interpolation w/ derivative
            if debug
                fprintf('Grad-Cubic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d],doPlot,0,t);
        elseif ~isLegal(g_prev)
            % Backtracking w/ quartic interpolation 3 points and derivative
            % of two
            if debug
                fprintf('Grad-Quartic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d; t_prev f_prev sqrt(-1)],doPlot,0,t);
        else
            % Backtracking w/ quintic interpolation of 3 points and derivative
            % of two
            if debug
                fprintf('Grad-Quintic BT\n');
            end
            t = polyinterp([0 f gtd; t f_new g_new'*d; t_prev f_prev g_prev'*d],doPlot,0,t);
         end
    end
    
    % Adjust if change in t is too small/large
    if t < temp*1e-3
        if debug
            fprintf('Interpolated Value Too Small, Adjusting\n');
        end
        t = temp*1e-3;
    elseif t > temp*0.6
        if debug
            fprintf('Interpolated Value Too Large, Adjusting\n');
        end
        t = temp*0.6;
    end

    % Store old point if doing three-point interpolation
    if LS_multi
        f_prev = f_new;
        t_prev = temp;
        if LS_interp == 2
            g_prev = g_new;
        end
    end
    
%     [f_new,g_new] = funObj(x + t*d);
    
    [f_new,g_new] = penalizedL2(x+t*d,data);

    funEvals = funEvals+1;

    % Check whether step size has become too small
    if max(abs(t*d)) <= progTol
        if debug
            fprintf('Backtracking Line Search Failed\n');
        end
        t = 0;
        f_new = f;
        g_new = g;
        break;
    end
end

% % Evaluate Hessian at new point
% if nargout == 6 && funEvals > 1 && saveHessianComp
%     [f_new,g_new,H] = funObj(x + t*d);
%     funEvals = funEvals+1;
% end

x_new = x + t*d;

end


%-------------------------------------------------------------------------%

function [nll,g,H] = penalizedL2(w,data)
% [nll,g,H] = penalizedL2(w,gradFunc,lambda,varargin)
% Adds L2-penalization to a loss function
% (you can use this instead of always adding it to the loss function code)

X = data.X;
yDummy = data.yDummy;
lambda = data.lambda;


% [nll,g] = gradFunc(w); 

[nll,g] = SoftmaxLossDummy(w,X,yDummy);

nll = nll+sum(lambda.*(w.^2)); %--L2 regu. NOTE: lambda(1)=0, so bias term is ignored from regu!
%-- add regu-term to g and H
if nargout > 1
    g = g + 2*lambda.*w;       %-- 1st derv, adding derv of  w'*w
end

if nargout > 2
    if isscalar(lambda)
        H = H + 2*lambda*eye(length(w));
    else
        H = H + diag(2*lambda);
    end
end
end

%-------------------------------------------------------------------------%


function [nll,g,H] = SoftmaxLossDummy(w,X,yDummy)
%-- Given w (param Theta of all classes), X (data) and ydummy (class label)
%-- Output: negLL, 1st derv g and 2nd derv H
%-- weights[nSmp 1]: optional (weight for each sample!)
%-- Call this func for Newton method 
%-- Eg: it is called in penalizedL2 to add L2-regularization!
%--     +   X[nSmp 1+nFea]: normX of 0-mean/1-std (except 1st bias:1-mean 0-std)
%--     +   w[nClass*param 1]: stack up params from all classes into 1 vector

nClass = size(yDummy,2);
[nSmp,nFea] = size(X);
w = reshape(w,[nFea nClass]); %-- back to matrix form: [1+nFea*nClass]

eta = X*w;              %-- [nSmp nClass] = [nSmp nFea]*[nFea nClass]
eta = eta - repmat(max(eta),size(eta,1),1); %-- avoid overflow--exp(.) too large
Z = sum(exp(eta), 2);   %-- [nSmp 1] Z(i)=sum_j exp(xi'*wj)
%-- logpred(i,k): xi'*wk - log(sum_j(exp(xi'*wj)))
logpred = eta - repmat(log(Z), 1, nClass);  %-- [nSmp nClass] 


ll = sum(logpred .* yDummy, 2);     %-- ll(i) = yik*xi'*wk - log(sum_j(exp(xi'*wj)))
nll = -sum(ll);          

%-- compute 1st derivative g=f'
if nargout > 1
  P = exp(logpred); %-- P(i,k) = P(k|xi) 
  % logpred(i,k)  = (xi'*wk) - log sum_j(exp(xi'*wj))
  % so exp(logpred(i,k)) = exp[(xi'*wk) - log sum_j(exp(xi'*wj)]
  %                      = exp(xi'*wk)/exp(log sum_j(exp(xi'*wj))
  %                      = exp(xi'*wk)/sum_j(exp(xi'*wj)) = p(k|i)
    g = zeros(nFea,nClass);
    for c = 1:nClass
        g(:,c) = -sum(X.*repmat(yDummy(:,c) - P(:,c),[1 nFea])); %-- f'= -X*(yc-pc)
    end
    g = reshape(g,[nFea*(nClass) 1]); %-- convert g to col-vec
end

%-- compute 2nd derivative H=f"
if nargout > 2
    H = zeros(nFea*(nClass));
    SM = P; 
    for c1 = 1:nClass
        for c2 = 1:nClass
            D = SM(:,c1).*((c1==c2)-SM(:,c2));
            H((nFea*(c1-1)+1):nFea*c1,(nFea*(c2-1)+1):nFea*c2) = X'*diag(sparse(D))*X;
        end
    end
end



end