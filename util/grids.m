funtion [ gXs, X ] = grids(X, grid_vec, numsteps)

    % Grid the whole space, which is different than gridding each
    % dimension. This gets VERY large VERY fast.

    % nargout 3 ==> grid the WHOLE SPACE, not just dimensions one by one.
    % This gets extremely large very fast and should only be used for
    % generating toy examples.
    %
    % If numsteps = 100 and D = 2, then gXs is a 10,000 x 2 matrix.
    % If numsteps = 100 and D = 3, then gXs is a 1,000,000 x 3 matrix.
    % If numsteps = 100 and D = 4, then gXs is a 100,000,000 x 4 matrix.
    % If numsteps = 100 and D = 5, then gXs is a 10,000,0000,000 x 5 matrix
    % 
    % If numsteps = 500 and D = 2, then gXs is a 250,000 x 2 matrix.
    % If numsteps = 500 and D = 3, then gXs is a 125,000,000 x 3 matrix.
    % If numsteps = 500 and D = 4, then gXs is a 62,500,000,000 x 4 matrix.
    %
    % If a dimension is held constant at its mean, then a column is added
    % by the number of rows would not be increased.
    
    gridX = cell(1, D);

    num_rows_gXs = numsteps^length(grid_vec); 
    % This is how long you need to make the num_rows_gXs x 1 column 
    % vector for dimensions held at their means.

    entryCounter = 1;
    for idx = 1:D
        if ismember(idx, grid_vec)
            % Grid the dimension to be plus / minus 2 standard 
            % deviations from max / min value the column data takes.
            lowerbound = min(X(:,idx)) - 2*sqrt(var(X(:,idx));
            upper_bound = max(X(:,idx)) + 2*sqrt(var(X(:,idx)));
            range = upper_bound - lower_bound;
            gXd = lower_bound:range/(numsteps-1):upper_bound;
            gridX{entryCounter} = gXd;
            entryCounter = entryCounter+1;
        end
    end

    gCopy = gridX;
    [gCopy{:}] = ndgrid(gridX{:});
    Xs = cell2mat(cellfun(@(m)m(:),gCopy,'UniformOutput',false));

    grelper = cell(1,D);
    for idx = 1:D
        if ismember(idx, grid_vec)
            grelper{idx} = 
        end
    end
    for tix = setdiff(1:D, grid_vec)
        % These are the indices that need to be inserted

    end

end