function [ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, interaction_indices)
% This function automates some of the prediction for generating marginal
% effects plots with plotme(). Grids are automatically generated for
% plotme() argument Xs given the covariate of interest d.
%       nargin 7 - No interaction effects are desired. A grid is defined
%       for covariate d alone, and all other covariates are held at their
%       mean.
%       nargin 8 - A vector of integers corresponding to covariate indices.
%       Must include d. Any dimension specified in interaction_indices will
%       be gridded. All other covariates are held at their mean.

    [~,D] = size(X);
    gridX = zeros(numsteps, D);
    for k = 1:D
        gridX(:,k) = mean(X(:,k))*ones(numsteps,1); % Hold all covariates at their respective mean
    end

    switch nargin 

        case 7 % When interaction_indices aren't specified, then only grid dimension d

            % Make the grid for dimension d
            range = max(X(:,d)) - min(X(:,d)) + 4*sqrt(var(X(:,d)));
            gXd = ((min(X(:,d)) - 2*sqrt(var(X(:,d)))):range/(numsteps-1):(max(X(:,d)) + 2*sqrt(var(X(:,d)))))';
            gridX(:,d) = gXd; % Overwrite d^th covariate with the grid

            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX); % Make the plot
        
        otherwise % The case when plotting interaction effects

            for k = 1:D
                gridX(:,k) = mean(X(:,k))*ones(numsteps,1); % Hold all covariates at their respective mean
            end
    
            % Make the grid over any index specified in interaction_indices
            for idx = interaction_indices
                range = max(X(:,idx)) - min(X(:,idx)) + 4*sqrt(var(X(:,idx)));
                gXd = ((min(X(:,idx)) - 2*sqrt(var(X(:,idx)))):range/(numsteps-1):(max(X(:,idx)) + 2*sqrt(var(X(:,idx)))))';
                gridX(:,idx) = gXd; % Overwrite d^th covariate with the grid
            end

            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX, interaction_indices); % Make the plot

    end

end

