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
d_me=2;
d_Xaxis=2;
numsteps = 100;
Xs = gridd(X, 2, numsteps); % Grid the first dimension

%
% gpd package
%

% Calculate sample marginal effects
[sme1, ~] = pme(hyp, meanfunc, covfunc, X, y);        
% Calculate marginal effects at unobserved locations
[pme1, pme2] = pme(hyp, meanfunc, covfunc, X, y, Xs);

% Define the credible region for predicted marginal effects
pred_f = [pme1(:,d_me)-1.96*sqrt(pme2(:,d_me)); flip(pme1(:,d_me)+1.96*sqrt(pme2(:,d_me)))]; 

% Plotting
hold on;
plt = fill([Xs(:,d_Xaxis); flip(Xs(:,d_Xaxis))], pred_f, [7 7 7]/8); % plot the credible region for predicted marginal effects
plot(X(:,d_Xaxis), sme1(:,d_me), 'o', 'Color', [1 0 0]) % plot sample marginal effects
plot(Xs(:,d_Xaxis), pme1(:,d_me), '.', 'Color', [0.9290 0.6940 0.1250]) % plot predicted marginal effects
hold off;

% Axis limits
xlim([min(Xs(:,d_Xaxis)), max(Xs(:,d_Xaxis))]) % Use the x limits of the errors bars
% Don't set a ylim

% Labeling and legend location
xlabel(strcat('X',num2str(d_Xaxis)))
ylabel(strcat('Marginal effect \partial Y \\ \partial X', num2str(d_me)))

% Legend location
legend('Location', 'southoutside');
legend('95% credible region for predictions', 'Sample marginal effects', 'Predicted marginal effects')




