function [ MEs, VARs ] = pme(hyp, meanfunc, covfunc, X, y, Xs)
% Return marginal effects and variances for row of the test inputs Xs.
% Include variances and 95% confidence intervals.
%       nargin 5 - Generate estimates with w.r.t. the training sample.
%       nargin otherwise - Generate estimates w.r.t. test points Xs 

    switch nargin
        case 5              
            Xs = X;         % If only 5 function inputs, then generate w.r.t. training sample and set Xs = X
    end

    [M,D] = size(Xs);       % [number of test points, number of covariates]
    MEs = zeros(D,M);       % Preallocate
    VARs = zeros(D,M);      % Preallocate
    for i = 1:M
        [ MEs(:,i), VARs(:,i) ] = me(hyp, meanfunc, covfunc, X, y, Xs(i,:));
    end

end