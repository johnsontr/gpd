% plot sandbox

% Make some data
N = 100;
D = 2;
X = normrnd(0,1,N,D);
sn = 0.1;
hyp.lik = log(sn);
u = normrnd(0,sn,N,1);
b0 = 1; b1 = 2; b2 = 3;
y = b0 * ones(N,1) + X * [b1 b2]' + u;

% GPR Model
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

% Learn GPR model parameter
hyp_iso = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, normalize(X), normalize(y));

% Make a grid
d=1;
numsteps = 100;
Xs = gridd(X, d, numsteps); % Grid the first dimension

%
% gpd package
%

[sme1, sme2] = pme(hyp, meanfunc, covfunc, X, y);        % sample
[pme1, pme2] = pme(hyp, meanfunc, covfunc, X, y, Xs);    % predictions 

% Make the error bars
plotSort = sortrows([Xs(:,d), pme1(:,d), pme2(:,d)], 1);
pred_f = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];

% Plotting
hold on;
plt = fill([plotSort(:,1); flip(plotSort(:,1))], pred_f, [7 7 7]/8);
plot(X(:,d), sme1(:,d), 'o')     % sample marginal effects
plot(Xs(:,d), pme1(:,d), '.')    % predicted marginal effects
hold off;

% Labeling
xlabel('X')
ylabel('Marginal effect \partial Y \\ \partial X')
xlim([min(Xs(:,d)), max(Xs(:,d))])
legend('95% credible region', ...
    'Sample marginal effects', ...
    'Predicted marginal effects')

