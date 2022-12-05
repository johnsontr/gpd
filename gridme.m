function plt = gridme(d, numsteps, hyp, meanfunc, covfunc, X, y)

    [~,D] = size(X);

    range = max(X(:,d)) - min(X(:,d)) + 4*sqrt(var(X(:,d)));
    gXd = ((min(X(:,d)) - 2*sqrt(var(X(:,d)))):range/(numsteps-1):(max(X(:,d)) + 2*sqrt(var(X(:,d)))))';
  
    gridX = zeros(numsteps, D);
    for k = 1:D
        gridX(:,k) = mean(X(:,k))*ones(numsteps,1); % Hold all covariates at their respective mean
    end
    gridX(:,d) = gXd; % Overwrite d^th covariate with the grid
    
    plt = plotme(d, hyp, meanfunc, covfunc, X, y, gridX);

end

