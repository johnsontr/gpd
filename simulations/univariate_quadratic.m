% Prepare workspace

clear;
clc;
close all;

%% Make DGP

N=100; % Number of observations in the training data.
D=1; % Number of covariates.
X = randn(N,1);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = -D; % For this example, the constant of regression is just negative the number of covariates.
b1 = 1; b2 = 2;
y = b0 * ones(N,1) + b1*X + b2*X.^2 + noise; % Marginal effects are b1 = 1, b2 = 2, ..., bD = D
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data

% True DGP: y = -1 + X + 2*X^2 + e;
% True ME: dy/dx = 1 + 4*X
% GPR with a single covariate.

%% covSEiso

% Specify the GPR model
meanfunc = {@meanZero};
covfunc = {@covSEiso};
ls = 1/2;
sf = 1;
hyp.cov = log([ ls sf ]);
hyp.mean = [];
likfunc = {@likGauss};
prior.lik = { {@priorTransform,@exp,@exp,@log,{@priorGamma,0.01,10}} }; % Gamma prior on the noise
prior.cov = { {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1}}, ...   % Gamma prior on ls
             {@priorTransform,@exp,@exp,@log,{@priorGamma,1,2}} };     % Gamma prior on sf
inffunc = {@infPrior, @infExact, prior}; 
p.method = 'LBFGS'; 
p.length = 100;

% Learn MAP parameter estimates
hyp_iso = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_X, train_y);

d=1;
numsteps=500;
univariate_quadratic_iso = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y);

% Save the grid plot
saveas(univariate_quadratic_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\univariate_quadratic_iso.png")
close;

%% covSEard

% covSEard is the same as covSEiso when there is only one predictor.