function [ mean_vec, diag_var_mat ] = me(hyp, meanfunc, covfunc, X, y, xs)
% Calculate the estimated marginal effect and variance for a specific test
% point xs. This is a support function for pme.
%       nargin 5 - Generate estimates with w.r.t. the training sample.
%       nargin otherwise - Generate estimates w.r.t. test points Xs   

% NOTE: Since this is built specifically for {@meanZero} Gaussian
% processes, the meanfunc function input is unused. The function input is 
% kept as a placeholder for when support is added for additional mean
% functions in the future.

    [ N, D ] = size(X);                                                 % [number of observations in the sample, number of covariates]
    
    % Make the diagonal matrix of the length scales, the form of which 
    % depends on whether the covariance function is covSEiso or covSEard.
    switch str2num(feval(covfunc{:}))
        case 2                                                          % If covfunc requires two inputs, then it's covSEiso
            Lambda = diag(repmat(exp(hyp.cov(1))^2,D,1)); 
        otherwise                                                       % If it's not covSEiso, then it's covSEard
            Lambda = diag((exp(hyp.cov(1:D)).^2));
    end
    
    % Make components used for calculation.
    Cs = (feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N));     % Define uninverted C(X,X) + sn^2 I for later use.

    dXs = zeros(N,D);                                                   % Preallocate
    for i = 1:N
        dXs(i,:) = xs - X(i,:);                                         % xs - xi is a 1xD vec
    end
    
    % Make partial derivative c(xs,X) partial xs, which is DxN
    d_c_xs_X_dxs = zeros(D,N);                                          % Preallocate
    for i = 1:N
        d_c_xs_X_dxs(:,i) = -(Lambda^-1) * (xs' - X(i,:)') * feval(covfunc{:}, hyp.cov, X(i,:), xs);
    end
    
    %
    %
    % Make the mean function and the covariance function from the components defined above.
    %
    %

    % mean function
    mean_vec = - Lambda^-1 * dXs' * (feval(covfunc{:}, hyp.cov, X, xs) .* (Cs \ y));

    % vcov function
    if str2num(feval(covfunc{:})) == 2
        var_mat = Lambda^-1 * exp(hyp.cov(2))^2 - d_c_xs_X_dxs * inv(Cs) * (-d_c_xs_X_dxs');
    else % If it's not covSEiso, then it's covSEard
        var_mat = Lambda^-1 * exp(hyp.cov(D+1))^2 - d_c_xs_X_dxs * inv(Cs) * (-d_c_xs_X_dxs');
    end
    
    % Only return the diagonals of var_mat, which corresponds to the
    % variance of the marginal distribution of the marginal effects.
    diag_var_mat = diag(var_mat);

end
