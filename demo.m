% MAP test

% This script trains a gp regression model to use for testing me, pme, and ame

N=50; % Number of observations in the training data.
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

% Unnormalized training, unnormalized post
learnedHyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, X, y);
[foo1, foo2] = pme(learnedHyp, meanfunc, covfunc, X, y);
mean(foo1')'
var(foo1')'
[mean(foo1')' - 1.96*sqrt(var(foo1')'), mean(foo1')' + 1.96*sqrt(var(foo1')')]

% Normalized training, unnormalized post
learnedHyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_X, train_y);
[foo1, foo2] = pme(learnedHyp, meanfunc, covfunc, X, y);
mean(foo1')'
var(foo1')'
[mean(foo1')' - 1.96*sqrt(var(foo1')'), mean(foo1')' + 1.96*sqrt(var(foo1')')]

% Normalized training and normalized post will generate garbage.

[doo1, doo2, doo3] = ame(learnedHyp, meanfunc, covfunc, X, y);
