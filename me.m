
% MEAN
% Dx1 vector

% VARIANCE
% 1. DxD
% 2. Dxn <--- same as (4) transposed
% 3. nxn
% 4. nxD <--- same as (2) transposed

n = size(X,1);

mean = foo;

MODE = 'iso';

% Switch length scale for the derivative depending on if the kernel is SEiso or SEard
LS_SWITCH = 0;
if strcmp(MODE,'iso')
    LS_SWITCH = 1; % For ISO
elseif strcmp(MODE,'ard')
    LS_SWITCH = d;
end

% Make the mean function
pd_cxsX_dxs = -1/exp(hyp.cov(LS_SWITCH))^2 * (xs(d) - X(:,d)) .* feval(covfunc{:}, hyp.cov, X, xs);
Cs = feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(n);
mean = pd_cxsX_dxs * Cs \ (y - feval(meanfunc{:}, hyp.mean, X));

% Make the covariance function

