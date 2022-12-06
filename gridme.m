function [ plt, gridX ] = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y, interaction_indices)

    % Future functionality: allow extra arguments to specify which
    % variables need to be gridded in case of interactions.

    [~,D] = size(X);

    gridX = zeros(numsteps, D);

    switch nargin 
        case 7

            for k = 1:D
                gridX(:,k) = mean(X(:,k))*ones(numsteps,1); % Hold all covariates at their respective mean
            end
    
            range = max(X(:,d)) - min(X(:,d)) + 4*sqrt(var(X(:,d)));
            gXd = ((min(X(:,d)) - 2*sqrt(var(X(:,d)))):range/(numsteps-1):(max(X(:,d)) + 2*sqrt(var(X(:,d)))))';
            gridX(:,d) = gXd; % Overwrite d^th covariate with the grid

            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX);
        
        otherwise

            for k = 1:D
                gridX(:,k) = mean(X(:,k))*ones(numsteps,1); % Hold all covariates at their respective mean
            end
    
            for idx = interaction_indices
                d = idx;
                range = max(X(:,d)) - min(X(:,d)) + 4*sqrt(var(X(:,d)));
                gXd = ((min(X(:,d)) - 2*sqrt(var(X(:,d)))):range/(numsteps-1):(max(X(:,d)) + 2*sqrt(var(X(:,d)))))';
                gridX(:,d) = gXd; % Overwrite d^th covariate with the grid
            end

            plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX, 1);

    end

end

