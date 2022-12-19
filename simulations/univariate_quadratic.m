% Prepare workspace

clear;
clc;
close all;
rng(1);

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

% Set parameters for plotting
d=1;
d_Xaxis = d;
numsteps=500;
Xs = gridd(X, d, numsteps);

% Generate marginal effects plot
univariate_quadratic_iso  = plotme(d, d_Xaxis, hyp_iso, meanfunc, covfunc, X, y, Xs);

% Add the true marginal effect to the plot
hold on;
ylim([-15 15]) % Set more reasonable y-axis limits
plot(Xs(:,d_Xaxis), b1+ (2*b2)*Xs(:,d_Xaxis), ':', 'Color', [0.8500 0.3250 0.0980],'LineWidth', 2, 'DisplayName', "True marginal effect");
plot(X(:,d_Xaxis), min(ylim) * ones(size(X(:,d_Xaxis),1)), '|', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'HandleVisibility','off') % Show observations
hold off;

% Save the grid plot
saveas(univariate_quadratic_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\univariate_quadratic_iso.png")
close;

%% covSEard

% covSEard is the same as covSEiso when there is only one predictor.
