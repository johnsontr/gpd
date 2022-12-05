function [ MEs, VARs ] = pme(hyp, meanfunc, covfunc, X, y, Xs)
% Return marginal effects and variances for row of the test inputs Xs.

    % If only 5 function inputs, then generate w.r.t. training sample.
    switch nargin
        case 5 
            Xs = X;
    end

    [M,D] = size(Xs);
    MEs = zeros(D,M);
    VARs = zeros(D,M);
    for i = 1:M
        [ MEs(:,i), VARs(:,i) ] = me(hyp, meanfunc, covfunc, X, y, Xs(i,:));
    end

end