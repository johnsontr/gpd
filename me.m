function [ mean_vec, var_vec ] = me(hyp, meanfunc, covfunc, X, y, xs)
% Calculate the estimated marginal effect and variance for a specific test
% point xs. This is a support function for pme.
%       nargin 5 - Generate estimates with w.r.t. the training sample.
%       nargin otherwise - Generate estimates w.r.t. test points Xs   

% NOTE: Since this is built specifically for {@meanZero} Gaussian
% processes, the meanfunc function input is unused. The function input is 
% kept as a placeholder for when support is added for additional mean
% functions in the future.

    [ N, D ] = size(X);
    
    % Make the length scale diagonal matrix, which depends on the covariance function.
    if str2num(feval(covfunc{:})) == 2                  % If the covfunc requires two inputs, then it's covSEiso
        Lambda = diag(repmat(exp(hyp.cov(1))^2,D,1));   % hyp.cov(2) is the scale factor
    else                                                % If it's not covSEiso, then it's covSEard
        Lambda = diag(exp(hyp.cov(1:D)).^2);             % hyp.cov(D+1) is the scale factor
    end
    
    % Make the total derivative c(xs,X) w.r.t. xs, which is a DxN matrix
    d_c_xs_X_dxs = zeros(D,N);
    for i = 1:N
        d_c_xs_X_dxs(:,i) = (Lambda^-1) * (X(i,:) - xs)' * feval(covfunc{:}, hyp.cov, X(i,:), xs);
    end

    % Make the total derivative of the mean at xs
    mean_vec = d_c_xs_X_dxs * ((feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N)) \ y);

    % The vcov matrix of the total derivative of the mean at xs
    if str2num(feval(covfunc{:})) == 2      % If covfunc = {@covSEiso}
        var_mat = (exp(hyp.cov(2))^2) * (Lambda^-1) - d_c_xs_X_dxs * inv(feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N)) * d_c_xs_X_dxs';
    else                                    % If covfunc = {@covSEard}
        var_mat = (exp(hyp.cov(D+1))^2) * (Lambda^-1) - d_c_xs_X_dxs * inv(feval(covfunc{:}, hyp.cov, X) + exp(hyp.lik)^2 * eye(N)) * d_c_xs_X_dxs';
    end
    
    % Only return the diagonals of var_mat, which corresponds to the
    % variance of the marginal distribution of the marginal effects.
    var_vec = diag(var_mat);

end
