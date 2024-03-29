function [ Xs, X ] = gridd(X, grid_vec, numsteps)

% Grid dimensions individually, not the whole space

    [~,D] = size(X);
    Xs = zeros(numsteps, D);

    for j = 1:D
        if ismember(j, grid_vec)
            % Grid the dimension to be plus / minus 2 standard 
            % deviations from max / min value the column data takes.
            lower_bound = min(X(:,j)) - 2*sqrt(var(X(:,j)));
            upper_bound = max(X(:,j)) + 2*sqrt(var(X(:,j)));
            range = upper_bound - lower_bound;
            gXd = lower_bound:range/(numsteps-1):upper_bound;
            Xs(:,j) = gXd;
        else 
            Xs(:,j) = repmat( mean(X(:,j)), 1, numsteps );
        end
    end
            
end

