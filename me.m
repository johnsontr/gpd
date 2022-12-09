function [ mean_vec, var_vec ] = me(hyp, meanfunc, covfunc, X, y, xs)
% Calculate the estimated marginal effect and variance for a specific test
% point xs. This is a support function for pme.
%       nargin 5 - Generate estimates with w.r.t. the training sample.
%       nargin otherwise - Generate estimates w.r.t. test points Xs   

% NOTE: Since this is built specifically for {@meanZero} Gaussian
% processes, the meanfunc function input is unused. The function input is 
% kept as a placeholder for when support is added for additional mean
% functions in the future.
    
    % First, make components common to either covfunc

    [ N, D ] = size(X);                                                 % [number of observations in the sample, number of covariates]

    Cs = (feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N));     % C(X,X) + sn^2 I    

    % Define the x-specific product-rule factor created from
    % differentiating a squared exponential kernel. This is common to both
    % covSEiso and covSEard
    dXs = zeros(N,D);                                                   % Preallocate
    for i = 1:N
        dXs(i,:) = xs - X(i,:);                                         % xs - xi is a 1xD vec
    end

    % Diagonal matrix of the length scales and the variance matrix of
    % marginal effects depend on whether the covariance function is covSEiso or covSEard.
    switch str2num(feval(covfunc{:}))

        case 2                                                          % If covfunc requires two inputs, then it's covSEiso

            % Diagonal matrix of the length scales
            Lambda = diag(repmat(exp(hyp.cov(1))^2,D,1)); 

            % Make partial derivative of c(X,xs) w.r.t. xs, which is NxD
            d_c_X_xs_dxs = -Lambda^-1 * dXs' * (feval(covfunc{:}, hyp.cov, X, xs)); 

            % Store the scale factor specific to the covariance function
            cov_scale_factor = exp(hyp.cov(2))^2;

        otherwise                                                       % If it's not covSEiso, then it's covSEard

            % Diagonal matrix of the length scales
            Lambda = diag((exp(hyp.cov(1:D)).^2)); 

            % Make partial derivative of c(X,xs) w.r.t. xs, which is NxD
            d_c_X_xs_dxs = -Lambda^-1 * dXs' * (feval(covfunc{:}, hyp.cov, X, xs)); 

            % Store the scale factor specific to the covariance function
            cov_scale_factor = exp(hyp.cov(D+1))^2;

    end

    % Marginal effect of the expected value of the Gaussian process at xs w.r.t. each covariate
    mean_vec = d_c_X_xs_dxs' .* (Cs \ y); 

    % Make the variance-covariance matrix of the marginal effects
    var_mat = cov_scale_factor * Lambda^-1 - d_c_X_xs_dxs' * inv(Cs) * (-d_c_X_xs_dxs); 

    % Only return the diagonals of var_mat, which corresponds to the variance of the marginal distribution of the marginal effects.
    var_vec = diag(var_mat); 

end
