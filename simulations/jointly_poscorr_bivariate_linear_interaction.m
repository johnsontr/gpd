% Prepare workspace

clear;
clc;
close all;

%% Make DGP

N=100; % Number of observations in the training data.
D=2; % Number of covariates.
mu = [0 0];
Sigma = [1 0.9; 0.9 1];
rng('default')  % For reproducibility
X = mvnrnd(mu,Sigma,N);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = -D; % For this example, the constant of regression is just negative the number of covariates.
b1 = 1; b2 = 2; b3 = 3;
y = b0 * ones(N,1) + X * [b1 b2]' + b3*(X(:,1) .* X(:,2))  + noise; % Marginal effects are b1 = 1, b2 = 2, ..., bD = D
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data

% True DGP: y = -2 + (X1) + 2*(X2) + 3*(X1*X2) + e;
% True ME for X1: dy/dx1 = 1 + 3*(X2)
% True ME for X2: dy/dx2 = 2 + 3*(X1)

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
[ jointly_poscorr_bivariate_linear_interaction_x1_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
plot(X(:,2), dydx(1,:), 'o', 'DisplayName', "Sample marginal effects")
plot(gridX(:,2), b1 + b3*gridX(:,2), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
xlabel('X2');
ylabel('Marginal effect \partial Y \\ \partial X1')
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-15 20])
plot(X(:,2), min(ylim) * ones(size(X(:,2),1)), '|');
hold off;


% Save the grid plot
saveas(jointly_poscorr_bivariate_linear_interaction_x1_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\jointly_poscorr_bivariate_linear_interaction_x1_iso.png")
close;

d=2;
numsteps=500;
[ jointly_poscorr_bivariate_linear_interaction_x2_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
plot(X(:,1), dydx(2,:), 'o', 'DisplayName', "Sample marginal effects")
plot(gridX(:,1), b2 + b3*gridX(:,1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
xlabel('X1');
ylabel('Marginal effect \partial Y \\ \partial X2')
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-15 20])
plot(X(:,1), min(ylim) * ones(size(X(:,1),1)), '|');
hold off;

% Save the grid plot
saveas(jointly_poscorr_bivariate_linear_interaction_x2_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\jointly_poscorr_bivariate_linear_interaction_x2_iso.png")
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

d=1;
numsteps=500;
[ jointly_poscorr_bivariate_linear_interaction_x1_ard, gridX ] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
plot(X(:,2), dydx(1,:), 'o', 'DisplayName', "Sample marginal effects")
plot(gridX(:,2), b1 + b3*gridX(:,2), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
xlabel('X2');
ylabel('Marginal effect \partial Y \\ \partial X1')
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-15 20])
plot(X(:,2), min(ylim) * ones(size(X(:,2),1)), '|');
hold off;


% Save the grid plot
saveas(jointly_poscorr_bivariate_linear_interaction_x1_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\jointly_poscorr_bivariate_linear_interaction_x1_ard.png")
close;

d=2;
numsteps=500;
[ jointly_poscorr_bivariate_linear_interaction_ard_x2, gridX ] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
plot(X(:,1), dydx(2,:), 'o', 'DisplayName', "Sample marginal effects")
plot(gridX(:,1), b2 + b3*gridX(:,1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
xlabel('X1');
ylabel('Marginal effect \partial Y \\ \partial X2')
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-15 20])
plot(X(:,1), min(ylim) * ones(size(X(:,1),1)), '|');
hold off;

% Save the grid plot
saveas(jointly_poscorr_bivariate_linear_interaction_ard_x2, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\jointly_poscorr_bivariate_linear_interaction_x2_ard.png")
close;