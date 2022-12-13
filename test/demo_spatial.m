%% prep workspace

clear;
clc;
close all;

%% Make spatial data

% Generate points on a square with y between pm 90 and x between pm 180

% This procedure is lazy. Plotting these points on a sphere will cluster at
% the poles. The stereographic projection distorts the poles, so the
% seeming randomness of this plot will disappear in an non-projected
% representation. If I really want data randomly generated on a sphere,
% then I need to generate it differently. There are a number of tutorials
% online, but this lazy method is fine to compare GPR and other methods.

N = 400;
lat_abs = 90; % Latitude is between -90 and 90 degrees
lon_abs = 180; % Longitude is between -180 and 180 degrees

lat = unifrnd(-lat_abs, lat_abs, N, 1);    
lon = unifrnd(-lon_abs, lon_abs, N, 1);  
loc = [lat, lon];

% plot(lon, lat, '.')
% xlim([-lon_abs,lon_abs])
% ylim([-lat_abs,lat_abs])

% Generate a spatial adjacency matrix. Use box contiguity.

% Define radius as a percentage of the latitude range
scl = 0.05;
radius = scl * 2*lat_abs; % 100*scl% of the range of latitude

% Make spatial adjacency matrices
[dub, stddub, notnb] = Wmaker(loc, radius);

% Histogram for how many neighbors each observation has.
% histogram(sum(dub))

%% Make the SAR DGP.

D=3; % Number of covariates.
X = normrnd(0,1,N,D);
sn = 0.1;
hyp.lik = log(sn);
noise = normrnd(0,sn,N,1);
b0 = 3; b1 = 1; b2 = 4; b3 = -2;
rho = 0.5; % Moderately strong positive spatial correlation
y = (eye(N) - rho*stddub) \ (b0 * ones(N,1) + X * [b1 b2 b3]' + noise);

% Normalize training data
train_X = normalize(X); % Normalize training data
train_y = normalize(y); % Normalize training data

% Plot the data
% plot(train_X(:,1),train_y, '.')
% plot(train_X(:,2),train_y, '.')
% plot(train_X(:,3),train_y, '.')

%% Direct and indirect effects

% Use LeSage and Pace (2009) definitions of direct and indirect effects

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
[ sar, gridX ] = gridme(d, numsteps, hyp_iso, meanfunc, covfunc, X, y, [1 2]);
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
% saveas(bivariate_linear_interaction_x1_iso, "C:\Users\johnsontr\Documents\GitHub\gpd\simulations\results\bivariate_linear_interaction_x1_iso.png")
% close;


