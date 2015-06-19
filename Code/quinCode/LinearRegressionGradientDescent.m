% Gradient descent algo for linear regression
% author: Nauman (recluze@gmail.com)
 
%set the data
X=[1 1 1 1 1 1 1; 22 49 80 26 40 54 91];
Y=[20 24 42 22 23 26 55];
hold on;
plot(X(2,:),Y, 'x');
 
% set the actual values of W
W = [5.775 0.474]';
YAct = (W' * X);
 
% GRADIENT DESCENT
W = zeros(2,1);     % initialize W to all zeros
m = size(X,2);      % number of instances
n = size(W,1);      % number of parameters
alpha = 0.000025;    % gradient descent step size
 
disp('Starting Weights:');
W
 
% 1) do the loop for W estimation using gradient descent ------------
for iter = 1 : 20 % let's just do 5 iterations for now ... not til convergence
% for each iteration
for i = 1 : n       % looping for each parameter W(i)
    % find the derivative of J(W) for all instances
    sum_j = 0;
    for j = 1 : m
        hW_j = 0;
        for inner_i = 1 : n
            hW_j = hW_j + W(inner_i) * X(inner_i,j);
        end
        sum_j = sum_j + ( (hW_j - Y(j)) * X(i,j) ) ;
    end
    % calculate new value of W(i)
    W(i) = W(i) - alpha * sum_j;
end
% plot the thingy
newY = (W' * X);
gColor = 0.05 * iter;
plot(X(2,:),newY, 'Color', [0 gColor 0]);
 
% end 1) ------------
end
% output the final weights
disp ('Final calculated weights');
W
% output actual weights from formula in Red W = [5.775 0.474]'
plot(X(2,:),YAct, 'Color',[1 0 0]);
% finish off
hold off;