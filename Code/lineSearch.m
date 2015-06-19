function alphak = lineSearch(f,d,x,rho,c)
% function alphak = linesearch(f,d,x,rho,c)
% Backtracking line search
% See Algorithm 3.1 on page 37 of Nocedal and Wright
% Input Parameters :
% f: MATLAB file that returns function value
% d: The search direction
% x: previous iterate
% rho :- The backtrack step between (0,1) usually 1/2
% c: parameter between 0 and 1 , usually 10^{-4}
% Output :
% alphak: step length calculated by algorithm
% Kartik's MATLAB code (27/1/08)

alphak = 1;
[fk, gk] = feval(f,x);
xx = x;
x = x + alphak*d;
fk1 = feval(f,x);
while fk1 > fk + c*alphak*(gk'*d),
  alphak = alphak*rho;
  x = xx + alphak*d;
  fk1 = feval(f,x);
end

