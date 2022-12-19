% Prepare workspace

clear;
clc;
close all;
rng(1);

%% Make DGP

N=100; % Number of observations in the training data.
D=2; % Number of covariates.
X = randn(N,D);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = -D; % For this example, the constant of regression is just negative the number of covariates.
b1 = 1; b2 = 2;
y = b0 * ones(N,1) + X * [b1 b2]' + noise; % Marginal effects are b1 = 1, b2 = 2, ..., bD = D
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data

% True DGP: y = -2 + (X1) + 2*(X2) + e;
% True ME for X1: dy/dx1 = 1
% True ME for X2: dy/dx2 = 2

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

% covSEiso d=1
d=1; % Set parameters for plotting
d_Xaxis = d;
numsteps=500;
Xs = gridd(X, d, numsteps);
bivariate_linear_x1_iso  = plotme(d, d_Xaxis, hyp_iso, meanfunc, covfunc, X, y, Xs); % Generate marginal effects plot
hold on; % Add the true marginal effect to the plot
ylim([-2 4]) % Set more reasonable y-axis limits
plot(Xs(:,d_Xaxis), b1*ones(size(Xs,1), 1), ':', 'Color', [0.8500 0.3250 0.0980],'LineWidth', 2, 'DisplayName', "True marginal effect");
plot(X(:,d_Xaxis), min(ylim) * ones(size(X(:,d_Xaxis),1)), '|', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'HandleVisibility','off') % Show observations
hold off;
% Save the grid plot
saveas(bivariate_linear_x1_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x1_iso.png")
close;

% covSEiso d=2
d=2; % Set parameters for plotting
d_Xaxis = d;
numsteps=500;
Xs = gridd(X, d, numsteps);
bivariate_linear_x2_iso  = plotme(d, d_Xaxis, hyp_iso, meanfunc, covfunc, X, y, Xs); % Generate marginal effects plot
hold on; % Add the true marginal effect to the plot
ylim([-2 4]) % Set more reasonable y-axis limits
plot(Xs(:,d_Xaxis), b2*ones(size(Xs,1), 1), ':', 'Color', [0.8500 0.3250 0.0980],'LineWidth', 2, 'DisplayName', "True marginal effect");
plot(X(:,d_Xaxis), min(ylim) * ones(size(X(:,d_Xaxis),1)), '|', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'HandleVisibility','off') % Show observations
hold off;
% Save the grid plot
saveas(bivariate_linear_x2_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x2_iso.png")
close;

%% covSEard

% Specify the GPR model
meanfunc = {@meanZero};
covfunc = {@covSEard};
ls = 1/2;
sf = 1;
hyp.cov = log([ ls*ones(D,1); sf ]);
hyp.mean = [];
likfunc = {@likGauss};
prior.lik = { {@priorTransform,@exp,@exp,@log,{@priorGamma,0.01,10}} }; % Gamma prior on the noise
for k = 1:D+1
    prior.cov{k} = {@priorTransform,@exp,@exp,@log,{@priorGamma,1,1}}; % Gamma prior on D length scales and 1 scale factor
end
inffunc = {@infPrior, @infExact, prior}; 
p.method = 'LBFGS'; 
p.length = 100;

% Learn MAP parameter estimates
hyp_ard = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, train_X, train_y);

% covSEard d=1
d=1; % Set parameters for plotting
d_Xaxis = d;
numsteps=500;
Xs = gridd(X, d, numsteps);
bivariate_linear_x1_ard  = plotme(d, d_Xaxis, hyp_ard, meanfunc, covfunc, X, y, Xs); % Generate marginal effects plot
hold on; % Add the true marginal effect to the plot
ylim([-2 4]) % Set more reasonable y-axis limits
plot(Xs(:,d_Xaxis), b1*ones(size(Xs,1), 1), ':', 'Color', [0.8500 0.3250 0.0980],'LineWidth', 2, 'DisplayName', "True marginal effect");
plot(X(:,d_Xaxis), min(ylim) * ones(size(X(:,d_Xaxis),1)), '|', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'HandleVisibility','off') % Show observations
hold off;
% Save the grid plot
saveas(bivariate_linear_x1_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x1_ard.png")
close;

% covSEard d=2
d=2; % Set parameters for plotting
d_Xaxis = d;
numsteps=500;
Xs = gridd(X, d, numsteps);
bivariate_linear_x2_ard  = plotme(d, d_Xaxis, hyp_ard, meanfunc, covfunc, X, y, Xs); % Generate marginal effects plot
hold on; % Add the true marginal effect to the plot
ylim([-2 4]) % Set more reasonable y-axis limits
plot(Xs(:,d_Xaxis), b2*ones(size(Xs,1), 1), ':', 'Color', [0.8500 0.3250 0.0980],'LineWidth', 2, 'DisplayName', "True marginal effect");
plot(X(:,d_Xaxis), min(ylim) * ones(size(X(:,d_Xaxis),1)), '|', 'Color', [0 0.4470 0.7410], 'LineWidth', 2, 'HandleVisibility','off') % Show observations
hold off;
% Save the grid plot
saveas(bivariate_linear_x2_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x2_ard.png")
close;

