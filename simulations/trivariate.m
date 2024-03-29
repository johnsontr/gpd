% Prepare workspace

clear;
clc;
close all;

%% Make DGP

% SEPARABLE AND NONSEPARABLE MODELS ONLY DIFFER BY THE OPERATION AFTER X1
% AND PARENTHESES AROUND X2 AND X3 TERMS.
%
% NONSEPARABLE: NOW THE SIGN OF X1 DEPENDS ON X2 AND X3. THE DIRECTION OF
% INFLUENCE OF EACH X-VARIABLE ON Y DEPENDS ON THE VALUES OF OTHER
% X-VARABLES. THIS CREATES LOCAL MARGINAL EFFECTS THAT DIFFER FROM GLOBAL
% MARGINAL EFFECTS.
%
% SEPARABLE
% SAR
%   y = inv(I - rho * W) * (3 + X1 + 4X2 - 2X3 + e)
% SEM
%   y = 3 + X1 + 4X2 - 2X3 + inv(I - rho * W) * e
%
% NONSEPARABLE
% SAR
%   y = inv(I - rho * W) * [3 + X1*(4X2 - 2X3) + e]
% SEM
%   y = 3 + X1*(4X2 - 2X3) + inv(I - rho * W) * e


% INSIDIOUS RANDOM EFFECTS
%
% SEPARABLE SAR
%   TRUE    y = inv(I - rho * W) * (a + X1 + 4X2 - 2X3 + e)
%           [a, x3] ~ MVN([0 0], [[1 rho],[rho 1]])

%           LeSage and Pace pp. 36 - 37
%           average direct impact: of own-change y_i from exp. var. x_r
%                                   = (1/n) sum(r=1,k)Sr(W)_ii
%           average indirect impact: (two and three are equivalent)

%           TRUE AVERDIRECT MARGINAL EFFECT OF X1:      1
%           TRUE INDIRECT MARGINAL EFFECT OF X1:    Sr(W)*X1
%   MODEL   y = inv(I - rho * W) * (a + b*X1 + c*X2 + e)
%   Random effects are jointly normal with an omitted variable X3 that
%   doesn't directly affect the estimate of X1 but will indirectly affect
%   the estimate through the simultaneity induced by the transformation
%   inv(I - rho * W).

% NONSEPARABLE SAR
%   TRUE    y = inv(I - rho * W) * (a + X1 * (4X2 - 2X3) + e)
%           [a, x3] ~ MVN([0 0], [[1 corr],[corr 1]])
%           TRUE DIRECT MARGINAL EFFECT OF X1:      4X2 - 2X3
%           TRUE INDIRECT MARGINAL EFFECT OF X1:    Sr(W)*X1

%   MODEL   y = inv(I - rho * W) * (a + b*X1 + c*X2 + e)
%   Random effects are jointly normal with an omitted variable X3 that
%   affects the estimate of X1.
%   Misspecifying or poorly estimating random effects can bias estimates
%   for coeffiients on X1.

% SAR Model notes
%
% y = sum(r=1,k) S_r (W) * X_r + inv(I - rho * W) * e
%       S_r (W) = inv(I - rho * W) beta_r
%           beta_r is a scalar
%           inv(I - rho * W) is an n x n matrix
%           X_r is the rth column vector of X
%           S_r (W) * X_r is an n x n matrix times an n x 1 column vector
%               So S_r (W) * X_r is an n x 1 vector

N=400; % Number of observations in the training data.
D=3; % Number of covariates.
X = normrnd(0,1,N,D);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = 3; b1 = 1; b4 = 2; b3 = -2;
y = b0 * ones(N,1) + X(:,1) * [b1 b2 b3]' + noise;
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data



% True DGP: y = 3 + (X1) * ( 4*(X2) - 2*(X3) ) + e;
% True ME for X1: dy/dx1 = 
% True ME for X2: dy/dx2 = 

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
[ bivariate_linear_interaction_x1_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp_iso, meanfunc, covfunc, X, y);            % sample
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
saveas(bivariate_linear_interaction_x1_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_interaction_x1_iso.png")
close;

d=2;
numsteps=500;
[ bivariate_linear_interaction_x2_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp_iso, meanfunc, covfunc, X, y);            % sample
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
saveas(bivariate_linear_interaction_x2_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_interaction_x2_iso.png")
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
[ bivariate_linear_interaction_x1_ard, gridX ] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp_ard, meanfunc, covfunc, X, y);            % sample
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
saveas(bivariate_linear_interaction_x1_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_interaction_x1_ard.png")
close;

d=2;
numsteps=500;
[ bivariate_linear_interaction_ard_x2, gridX ] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y, [1 2]);
hold on;
[dydx, ~] = pme(hyp_ard, meanfunc, covfunc, X, y);            % sample
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
saveas(bivariate_linear_interaction_ard_x2, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_interaction_x2_ard.png")
close;