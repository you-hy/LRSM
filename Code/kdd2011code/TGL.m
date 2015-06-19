function [x, funVal, ValueL]=TGL(A, y, z, opts)
%% Function TGL:
%      Least Squares Loss for Multi-task Learning
%             via the (group) L1/Lq-norm Regularization
%                 and Temporal Smoothness Regularization
%
%% Problem
%
%  min  1/2 sum_i || A_i x_i - y_i||^2 + z * sum_j ||x^j||_q
%                                      + z1 * 1/2  ||X||_F^2
%                                      + z2 * 1/2 ||X H||_F^2
%
%  x^j denotes the j-th row of x
%  x_i denotes the i-th column of x
%  y_i denotes the i-th column of y
%
%  For the case that the multi tasks share the same data
%  matrix, please refer to the functions:
%            mcLeastR and mcLogisticR.
%
%% Input parameters:
%
%  A-         Matrix of size m x n
%                A can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  y -        Response vector (of size m x 1)
%  z -        L1/Lq norm regularization parameter (z >=0)
%  opts-      Optional inputs (default value: opts=[])
%
%% Output parameters:
%  x-         Solution (of size n x k)
%  funVal-    Function value during iterations
%
%% Copyright (C) 2009-2011 Jiayu Zhou, Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jiayu Zhou via Jiayu.Zhou@asu.edu
%
% Last modified on Jan 19, 2011.
%
%% Related papers
%
% [1]  Jun Liu, Shuiwang Ji, and Jieping Ye, Multi-Task Feature Learning
%      Via Efficient L2,1-Norm Minimization, UAI, 2009
%
% [2]  Jun Liu, Lei Yuan, Songcan Chen and Jieping Ye, Multi-Task Feature Learning
%      Via Efficient L2,1-Norm Minimization, Technical Report ASU, 2009.
%
%% Related functions:
%
%  sll_opts, initFactor, pathSolutionLeast
%  mtLogisticR, eppVectorR
%
%%

%% Verify and initialize the parameters
%%
if (nargin <4)
    error('\n Inputs: A, y, z, and opts.ind should be specified!\n');
end

[m,n]=size(A);

if (length(y) ~=m)
    error('\n Check the length of y!\n');
end

if (z<0)
    error('\n z should be positive!\n');
end

opts=sll_opts(opts); % run sll_opts to set default values (flags)

%% Detailed initialization
%%



% Initialize ind and q
if ~isfield(opts,'ind')
    error('\n In mtLeastR, .ind should be specified');
else
    ind=opts.ind;
    k=length(ind)-1;
    
    if ind(k+1)~=m
        error('\n Check opts.ind');
    end
end

% Verify the input of H and z2

% Initialize H
if ~isfield(opts,'H')
    %error('\n In this function, .H should be specified');
    %by default the H is 1, -1. 
    H=zeros(k,k-1);
    H(1:(k+1):end)=1;
    H(2:(k+1):end)=-1;
else
    H=opts.H;
    if (size(H,1)~=k || size(H,2)~=k-1)
        error('\n Check the dimensionality of opts.H! It should be %d x %d',k,k-1);
    end
end

% Initialize z1
if ~isfield(opts,'z1')
    error('\n In this function, .z1 should be specified');
else
    z1=opts.z1;
end

% Initialize z2
if ~isfield(opts,'z2')
    error('\n In this function, .z2 should be specified');
else
    z2=opts.z2;
end

% Initialize q
% setting q>=1e6 results infinity norm.
if (~isfield(opts,'q'))
    q=2; opts.q=2;
else
    q=opts.q;
    if (q<1)
        error('\n q should be larger than 1');
    end
end

%% Normalization

% Please refer to sll_opts for the definitions of mu, nu and nFlag
%
% If .nFlag =1, the input matrix A is normalized to
%                     A= ( A- repmat(mu, m,1) ) * diag(nu)^{-1}
%
% If .nFlag =2, the input matrix A is normalized to
%                     A= diag(nu)^{-1} * ( A- repmat(mu, m,1) )
%
% Such normalization is done implicitly
%     This implicit normalization is suggested for the sparse matrix
%                                    but not for the dense matrix
%

if (opts.nFlag~=0)
    if (isfield(opts,'mu'))
        mu=opts.mu;
        if(size(mu,2)~=n)
            error('\n Check the input .mu');
        end
    else
        mu=mean(A,1);
    end
    
    if (opts.nFlag==1)
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=n)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,1)/m).^(0.5); nu=nu';
        end
    else % .nFlag=2
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=m)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,2)/n).^(0.5);
        end
    end
    
    ind_zero=find(abs(nu)<= 1e-10);    nu(ind_zero)=1;
    % If some values in nu is typically small, it might be that,
    % the entries in a given row or column in A are all close to zero.
    % For numerical stability, we set the corresponding value to 1.
end

if (~issparse(A)) && (opts.nFlag~=0)
    fprintf('\n -----------------------------------------------------');
    fprintf('\n The data is not sparse or not stored in sparse format');
    fprintf('\n The code still works.');
    fprintf('\n But we suggest you to normalize the data directly,');
    fprintf('\n for achieving better efficiency.');
    fprintf('\n -----------------------------------------------------');
end

%% Starting point initialization

ATy=zeros(n, k);
% compute AT y
for i=1:k
    ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
    
    if (opts.nFlag==0)
        tt =A(ind_i,:)'*y(ind_i,1);
    elseif (opts.nFlag==1)
        tt= A(ind_i,:)'*y(ind_i,1) - sum(y(ind_i,1)) * mu';
        tt=tt./nu(ind_i,1);
    else
        invNu=y(ind_i,1)./nu(ind_i,1);
        tt=A(ind_i,:)'*invNu - sum(invNu)*mu';
    end
    
    ATy(:,i)= tt;
end

% process the regularization parameter
if (opts.rFlag==0)
    lambda=z;
else % z here is the scaling factor lying in [0,1]
    if (z<0 || z>1)
        error('\n opts.rFlag=1, and z should be in [0,1]');
    end
    
    if q==1
        q_bar=Inf;
    elseif q>=1e6
        q_bar=1;
    else
        q_bar=q/(q-1);
    end
    lambda_max=0;
    for i=1:n
        lambda_max=max(lambda_max,...
            norm(  ATy(i,:), q_bar) );
    end
    lambda=z*lambda_max;
end

% initialize a starting point
if opts.init==2
    x=zeros(n,k);
else
    if isfield(opts,'x0')
        x=opts.x0;
        if ( size(x,1)~=n || size(x,2)~=k )
            error('\n Check the input .x0');
        end
    else
        x=ATy;  % if .x0 is not specified, we use ratio*ATy,
        % where ratio is a positive value
    end
end

Ax=zeros(m,1);
% compute Ax: Ax_i= A_i * x_i
for i=1:k
    ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
    m_i=ind(i+1)-ind(i);          % number of samples in the i-th group
    
    if (opts.nFlag==0)
        Ax(ind_i,1)=A(ind_i,:)* x(:,i);
    elseif (opts.nFlag==1)
        invNu=x(:,i)./nu; mu_invNu=mu * invNu;
        Ax(ind_i,1)=A(ind_i,:)*invNu -repmat(mu_invNu, m_i, 1);
    else
        Ax(ind_i,1)=A(ind_i,:)*x(:,i)-repmat(mu*x(:,i), m, 1);
        Ax(ind_i,1)=Ax./nu(ind_i,1);
    end
end

if (opts.init==0) % If .init=0, we set x=ratio*x by "initFactor"
    % Please refer to the function initFactor for detail
    
    x_norm=0;
    for i=1:n
        
        if (q>=1e6)
            x_norm=x_norm+ norm( x(i,:), inf );
        else
            x_norm=x_norm+ norm( x(i,:), q );
        end
        
    end
    
    if x_norm>=1e-6
        ratio=initFactor(x_norm, Ax, y, lambda,'mtLeastR');
        x=ratio*x;    Ax=ratio*Ax;
    end
end

%% The main program

bFlag=0; % this flag tests whether the gradient step only changes a little

L=1;
% We assume that the maximum eigenvalue of A'A is over 1

% assign xp with x, and Axp with Ax
xp=x; Axp=Ax; xxp=zeros(n,k);

alphap=0; alpha=1;

for iterStep=1:opts.maxIter
    % --------------------------- step 1 ---------------------------
    % compute search point s based on xp and x (with beta)
    beta=(alphap-1)/alpha;    s=x + beta* xxp;
    
    % --------------------------- step 2 ---------------------------
    % line search for L and compute the new approximate solution x
    
    % compute the gradient (g) at s
    As=Ax + beta* (Ax-Axp);
    
    % compute ATAs : n x k
    for i=1:k
        ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
        
        if (opts.nFlag==0)
            tt =A(ind_i,:)'*As(ind_i,1);
        elseif (opts.nFlag==1)
            tt= A(ind_i,:)'*As(ind_i,1) - sum(As(ind_i,1)) * mu';
            tt=tt./nu(ind_i,1);
        else
            invNu=As(ind_i,1)./nu(ind_i,1);
            tt=A(ind_i,:)'*invNu - sum(invNu)*mu';
        end
        
        ATAs(:,i)= tt;
    end
    
    % obtain the gradient g
    g=ATAs-ATy;
    
    % add extra smooth parts
    g=g + z1 * s  + z2* s * H * H';
    
    % copy x and Ax to xp and Axp
    xp=x;    Axp=Ax;
    
    while (1)
        % let s walk in a step in the antigradient of s to get v
        % and then do the L1/Lq-norm regularized projection
        v=s-g/L;
        
        % L1/Lq-norm regularized projection
        x=eppMatrix(v, n, k, lambda/ L, q);
        
        v=x-s;  % the difference between the new approximate solution x
        % and the search point s
        
        % compute Ax: Ax_i= A_i * x_i
        for i=1:k
            ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
            m_i=ind(i+1)-ind(i);          % number of samples in the i-th group
            
            if (opts.nFlag==0)
                Ax(ind_i,1)=A(ind_i,:)* x(:,i);
            elseif (opts.nFlag==1)
                invNu=x(:,i)./nu; mu_invNu=mu * invNu;
                Ax(ind_i,1)=A(ind_i,:)*invNu -repmat(mu_invNu, m_i, 1);
            else
                Ax(ind_i,1)=A(ind_i,:)*x(:,i)-repmat(mu*x(:,i), m, 1);
                Ax(ind_i,1)=Ax./nu(ind_i,1);
            end
        end
        
        Av=Ax -As;
        r_sum=norm(v,'fro')^2; l_sum=Av'*Av + z1 * norm(v, 'fro')^2 + z2 * norm(v*H,'fro')^2;
        
        % we need to revise here for checking the termination
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        % the condition is ||Av||_2^2 <= L * ||v||_2^2
        if(l_sum <= r_sum * L)
            break;
        else
            L=max(2*L, l_sum/r_sum);
            % fprintf('\n L=%5.6f',L);
        end
    end
    
    % --------------------------- step 3 ---------------------------
    % update alpha and alphap, and check whether converge
    alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;
    
    ValueL(iterStep)=L;
    
    xxp=x-xp;   Axy=Ax-y;
    funVal(iterStep)=Axy'*Axy/2;
    
    for i=1:n
        if (q>=1e6)
            funVal(iterStep)=funVal(iterStep)+ lambda* norm(  x(i,:), inf);
        else
            funVal(iterStep)=funVal(iterStep)+ lambda* norm(  x(i,:), q);
        end
        
    end
    
    % add a new term to the function value
    funVal=funVal + 0.5 * z1 * norm(x,'fro')^2 + 0.5 * z2*  norm(x*H, 'fro')^2;
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    switch(opts.tFlag)
        case 0
            if iterStep>=2
                if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iterStep>=2
                if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                        opts.tol* funVal(iterStep-1))
                    break;
                end
            end
        case 2
            if ( funVal(iterStep)<= opts.tol)
                break;
            end
        case 3
            norm_xxp=norm(xxp,'fro');
            if ( norm_xxp <=opts.tol)
                break;
            end
        case 4
            norm_xp=norm(xp,'fro');    norm_xxp=norm(xxp,'fro');
            if ( norm_xxp <=opts.tol * max(norm_xp,1))
                break;
            end
        case 5
            if iterStep>=opts.maxIter
                break;
            end
    end
end

