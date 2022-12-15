function [ W, stdW, no_neighbor_indices ] = Wmaker(loc, radius)
% This function takes in a matrix of N lat/lon coordinates and generates a
% spatial contiguity matrix for neighbors within radius distance.
% The function returns:
%   1. the raw matrix W with i,j entry being 0 / 1 if not neighbors / neighbors and
%   2. the row standardized W
%   3. a vector of the observations that have no neighbors

    [N,~] = size(loc);

    % For any given point, generate neighbors.
    W = zeros(N);
    for i = 1:N
        for not_i = 1:i
            if not_i == i
                continue
            else
                if (loc(not_i,1)-loc(i,1))^2 + (loc(not_i,2)-loc(i,2))^2 < radius^2
                    W(i,not_i)=1;
                    W(not_i,i)=1;
                end
            end
        end
    end
    
    % Which observations have no neighbors?
    counter = 1;
    no_neighbor_indices = [];
    for i = 1:N
        if sum(W(i,:)) == 0
            no_neighbor_indices(counter) = i;
            counter = counter+1;
        else
            continue
        end
    end
    
    % Row standardize the matrix
    stdW = W;
    for i = 1:N
        scale = sum(stdW(i,:));
        if scale == 0
            continue
        else
            for j = 1:N
                stdW(i,j) = stdW(i,j) / scale;
            end
        end
    end

end