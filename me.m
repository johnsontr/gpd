
function [ mean_vec, var_mat ] = me(hyp, meanfunc, covfunc, X, y, xs)

% Build the mean and variance of the GPME. Output should be a Dx1 vector
% and a DxD matrix for any input xs. Then call this function to generate on
% the sample or on a predict plot.

% input: hyp, meanfunc, covfunc, X, y, xs
%   X and y must be normalized.

% Add a "data=" option where certain things are calculated automatically.
% Split a grid automatically with a reasonable number of points based on
% the range.


%%%
% Support variables for testing
N=50; % Number of observations in the training data.
D=2; % Number of covariates.
X = randn(N,D);
xs = randn(1,D);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = -D; % For this example, the constant of regression is just negative the number of covariates.
y = b0 .* ones(N,1) + X * (1:D)' + noise; % Marginal effects are b1 = 1, b2 = 2, ..., bD = D
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data
meanfunc = {@meanZero};
covfunc = {@covSEiso};
ls = 1;
sf = 1;
hyp.cov = log([ ls sf ]);
hyp.mean = [];
%likfunc = {@likGauss};
%inffunc = {}





[ N, D ] = size(X);

% For meanfunc = {@meanZero}
% Partial E[ ys | X, y, xs] / Partial xs = Partial c(xs, X) / Partial xs * inv(C(X) + sn^2 I) * y

% Define components

if str2num(feval(covfunc{:})) == 2
    Lambda = diag(repmat(exp(hyp.cov(1))^2,D,1)); 
else % If it's not covSEiso, then it's covSEard
    Lambda = diag(exp(hyp.cov(1:D-1))^2); 
end


% Define uninverted C(X) + sn^2 I
Cs = (feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N));

% Define xs - xi, a 1xD vec, make n x D
dXs = zeros(N,D);
for i = 1:N
    dXs(i,:) = xs - X(i,:);
end

dc_xs_X_dxs = zeros(D,N);
for i = 1:N
    dc_xs_X_dxs(:,i) = -(Lambda^-1) * (xs' - X(i,:)') * feval(covfunc{:}, hyp.cov, X(i,:), xs);
end

% specific to the covSEiso kernel
% mean
mean_vec = - Lambda^-1 * dXs' * (feval(covfunc{:}, hyp.cov, X, xs) .* (Cs \ y));
% vcov
if str2num(feval(covfunc{:})) == 2
    var_mat = Lambda^-1 * exp(hyp.cov(2))^2 - dc_xs_X_dxs * inv(Cs) * (-dc_xs_X_dxs');
else % If it's not covSEiso, then it's covSEard
    var_mat = Lambda^-1 * exp(hyp.cov(D+1))^2 - dc_xs_X_dxs * inv(Cs) * (-dc_xs_X_dxs');
end



 
% MODE = 'iso';
% % Switch length scale for the derivative depending on if the kernel is SEiso or SEard
% LS_SWITCH = 0;
% if strcmp(MODE,'iso')
%     LS_SWITCH = 1; % For ISO
% elseif strcmp(MODE,'ard')
%     LS_SWITCH = d;
% end
% 
% % Make the mean function
% pd_cxsX_dxs = -1/exp(hyp.cov(LS_SWITCH))^2 * (xs(d) - X(:,d)) .* feval(covfunc{:}, hyp.cov, X, xs);
% Cs = feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(n);
% mean = pd_cxsX_dxs * Cs \ (y - feval(meanfunc{:}, hyp.mean, X));
% 


end
