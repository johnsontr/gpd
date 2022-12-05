% Train a zero mean, covSEard GPR model on a simple DGP for testing

close all;
clear;
clc;

%% Define the data generating process

N=200; % Number of observations in the training data.
M=100; % Number of prediction points in the test data.
D=2; % Number of covariates.
X = randn(N,D);
xs = randn(1,D);
Xs = randn(M,D);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = -D; % For this example, the constant of regression is just negative the number of covariates.
y = b0 .* ones(N,1) + X * (1:D)' + noise; % Marginal effects are b1 = 1, b2 = 2, ..., bD = D
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data

%% Define the GPR model

meanfunc = {@meanZero};
covfunc = {@covSEard};
ls = 1/2;
sf = 1;
hyp.cov = log([ ls*ones(1,D) sf ]);
hyp.mean = [];
likfunc = {@likGauss};
prior.lik = { {@priorTransform,@exp,@exp,@log,{@priorGamma,0.01,10}} }; % Gamma prior on the noise
for k = 1:D
    prior.cov{k} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1}};  % Gamma prior on ls
end
prior.cov{D+1} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,2}};    % Gamma prior on sf
inffunc = {@infPrior, @infExact, prior}; 
p.method = 'LBFGS'; 
p.length = 100;

%% Learn MAP parameter estimates

learnedHyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_X, train_y);

%% Verify pme and ame functionality

[f1, f2] = pme(learnedHyp, meanfunc, covfunc, X, y);
[g1, g2] = pme(learnedHyp, meanfunc, covfunc, X, y, Xs);
[h1, h2, h3] = ame(learnedHyp, meanfunc, covfunc, X, y);
[i1, i2, i3] = ame(learnedHyp, meanfunc, covfunc, X, y, Xs);

%% Verify plotme functionality

% w.r.t. Sample X
d=1;
plotme(d, hyp, meanfunc, covfunc, X, y);
d=2;
plotme(d, hyp, meanfunc, covfunc, X, y);

% w.r.t. Predictions Xs
d=1;
plotme(d, hyp, meanfunc, covfunc, X, y, Xs);
d=2;
plotme(d, hyp, meanfunc, covfunc, X, y, Xs);

