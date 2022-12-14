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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numsteps = 100;

[~,D] = size(X);
gridX = cell(1, D);

% Grid each covariate separately
for idx = 1:D
    range = max(X(:,idx)) - min(X(:,idx)) + 4*sqrt(var(X(:,idx)));
    gXd = ((min(X(:,idx)) - 2*sqrt(var(X(:,idx)))):range/(numsteps-1):(max(X(:,idx)) + 2*sqrt(var(X(:,idx)))))';
    gridX{idx} = gXd; % Overwrite d^th covariate with the grid
end

% Make combined grid
gCopy = gridX;
[gCopy{:}] = ndgrid(gridX{:});
Xs = cell2mat(cellfun(@(m)m(:),gCopy,'UniformOutput',false));


% Call pme
tic;
[f1, f2] = pme(hyp_iso, meanfunc, covfunc, X, y);            % sample
[voot1, voot2, voot3] = ame(hyp_iso, meanfunc, covfunc, X, y); 
toc;

% Xs is 10,000 x 2 when numsteps = 100
tic;
[g1, g2] = pme(hyp_iso, meanfunc, covfunc, X, y, Xs);       % predictions     
toc;


% For each unique value in the X1 dimension, take the mean of all
% corresponding g1 values. Calculate the variance. This is the estimate for
% the marginal effect at the unique value in X1 as X2 ranges over its grid.

% For calc

test = [Xs(:,1), g1(:,1)];
voot = sortrows(test, 1);

for i = 1:numsteps
    v(i) = mean(voot(((i-1)*100 + 1):(i*100), 2));
    bee(i) = var(voot(((i-1)*100 + 1):(i*100), 2));
    xx(i) = voot((i-1)*100+1, 1);
end
z = [xx,v];

hold on;
plot(z(:,1), z(:,2), '.')
plot(z(:,1), bee, 'o')
foo = ylim;
plot(X(:,1), foo(1)*zeros(size(X(:,1),1),1), '|')
hold off;

%
%
%

d=1;

plotSort = sortrows([Xs(:,d), g1(:,d), g2(:,d)], 1);
g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
plt = fill([Xs(:,1); flip(Xs(:,1))], g, [7 7 7]/8);



g = [g1(:,d)+1.96*sqrt(g2(:,d)); flip( g1(:,d)-1.96*sqrt(g2(:,d)) )];
hold on;
plt = fill([Xs(:,1); flip(Xs(:,1))], g, [7 7 7]/8);

g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
hold on;
plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
plot(X(:,d), f1(:,d), 'o')                             % sample
plot(Xs(:,d), g1(:,d), '.')                            % predictions
hold off;
xlabel('X')
ylabel('Marginal effect \partial Y \\ \partial X')
xlim([min(Xs(:,d)), max(Xs(:,d))])
legend('95% credible region', ...
    'Sample marginal effects', ...
    'Predicted marginal effects')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




d=1;
numsteps=100;
[ bivariate_linear_x1_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y);
hold on;
plot(gridX(:,d), b1*ones(size(gridX,1), 1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-2 4])
plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), '|');
hold off;


% Save the grid plot
saveas(bivariate_linear_x1_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x1_iso.png")
close;

d=2;
numsteps=100;
[ bivariate_linear_x2_iso, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y);
hold on;
plot(gridX(:,d), b2*ones(size(gridX,1), 1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-2 4])
plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), '|');
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

d=1;
numsteps=500;
[ bivariate_linear_x1_ard, gridX ] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y);
hold on;
plot(gridX(:,d), b1*ones(size(gridX,1), 1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-2 4])
plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), '|');
hold off;

% Save the grid plot
saveas(bivariate_linear_x1_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x1_ard.png")
close;

d=2;
numsteps=500;
[ bivariate_linear_x2_ard, gridX] = gridme(d, numsteps, hyp_ard, meanfunc, covfunc, X, y);
hold on;
plot(gridX(:,d), b2*ones(size(gridX,1), 1), ':', 'LineWidth', 2, 'DisplayName', "True marginal effect");
legend('Location', 'southoutside');
legend('AutoUpdate', 'off');
ylim([-2 4])
plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), '|');
hold off;

% Save the grid plot
saveas(bivariate_linear_x2_ard, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_x2_ard.png")
close;