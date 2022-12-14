function [ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, plot_Xdim)
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
    for idx = 1:D    % Grid everything
        range = max(X(:,idx)) - min(X(:,idx)) + 4*sqrt(var(X(:,idx)));
        gXd = ((min(X(:,idx)) - 2*sqrt(var(X(:,idx)))):range/(numsteps-1):(max(X(:,idx)) + 2*sqrt(var(X(:,idx)))))';
        gridX(:,idx) = gXd; % Overwrite d^th covariate with the grid
    end

    

    % This gridding is insufficient. For any observations of dimension d, I
    % need a grid across another dimension d'. For any observation of d and
    % a particular observation on the grid d', I need another grid across
    % dimension d''. 

    switch nargin 
        case 7 % If plot_Xdim isn't specified, plot X dim is dimension d
            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX);
        case 8 % Grid everything
            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX, plot_Xdim);
        %case 9 % Only grid certain dimensions
    end

end

