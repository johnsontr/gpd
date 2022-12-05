function [gmm_mean, gmm_mean_var, cred95] = ame(hyp, meanfunc, covfunc, X, y, Xs)

    % If only 5 function inputs, then generate w.r.t. training sample.
    switch nargin
        case 5 
            [MEs, ~] = pme(hyp, meanfunc, covfunc, X, y);
        otherwise
            [MEs, ~] = pme(hyp, meanfunc, covfunc, X, y, Xs);
    end
    
    gmm_mean = mean(MEs')';
    gmm_mean_var = var(MEs')';

    cred95 = [gmm_mean - 1.96*sqrt(gmm_mean_var), gmm_mean + 1.96*sqrt(gmm_mean_var)];
    %rowNames = {'a','b','c'};
    %colNames = {'x','y','z'};
    %sTable =
    %array2table(sample,'RowNames',rowNames,'VariableNames',colNames);

end

