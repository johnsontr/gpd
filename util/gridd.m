function [ Xs, X ] = gridd(X, grid_vec, numsteps)

% Grid dimensions individually, not the whole space

    [~,D] = size(X);
    gridX = zeros(numsteps, D);

    for j = 1:D
        if ismember(j, grid_vec)
            % Grid the dimension to be plus / minus 2 standard 
            % deviations from max / min value the column data takes.
            lower_bound = min(X(:,idx)) - 2*sqrt(var(X(:,idx));
            upper_bound = max(X(:,idx)) + 2*sqrt(var(X(:,idx)));
            range = upper_bound - lower_bound;
            gXd = lower_bound:range/(numsteps-1):upper_bound;
            gridX(:,j) = gXd;
        else 
            gridX(:,j) = repmat( mean(X(:,j)), 1, D );
        end
    end
            
end

