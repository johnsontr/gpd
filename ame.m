function [gmm_mean, gmm_mean_var, cred95] = ame(hyp, meanfunc, covfunc, X, y, Xs)
% Return the mean of sample marginal effects calculated by calling pme().
% Include variances and 95% confidence intervals.
%       nargin 5 - Generate estimates with w.r.t. the training sample.
%       nargin otherwise - Generate estimates w.r.t. test points Xs

    switch nargin
        case 5 
            [MEs, ~] = pme(hyp, meanfunc, covfunc, X, y);       % If 5 function inputs, then generate w.r.t. training sample and set Xs = X.
        otherwise
            [MEs, ~] = pme(hyp, meanfunc, covfunc, X, y, Xs);   % If 6 function inputs, then generate w.r.t. test points Xs.
    end
    
    gmm_mean = mean(MEs')';     % Sample mean
    gmm_mean_var = var(MEs')';  % Sample variance
    cred95 = [gmm_mean - 1.96*sqrt(gmm_mean_var), gmm_mean + 1.96*sqrt(gmm_mean_var)]; % 95% confidence interval

end

