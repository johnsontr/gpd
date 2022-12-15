function plt = plotme(d, d_plot, hyp, meanfunc, covfunc, X, y, Xs)
% This function generates marginal effect plots.
%       nargin 7 - If Xs is provided but an eighth argument is not, then
%       predicted marginal effects as well as sample marginal effects are
%       plotted.
%       nargin 8 - If eight arguments are provided, then the plot is for
%       marginal effects when interactions are suspected, so some of the
%       plot elements from nargin 7 are excluded so the user can specify
%       them. For example, sample marginal effects, xlabel, and ylabel are
%       not included.

    switch nargin

        case 7 % When plot_Xdim isn't specified
            [f1, f2] = pme(hyp, meanfunc, covfunc, X, y);        % sample
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);    % predictions     


            plotSort = sortrows([Xs(:,d), g1(d,:)', g2(d,:)'], 1);

            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];

            hold on;
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            plot(X(:,d), f1(d,:)', 'o')     % sample marginal effects
            plot(Xs(:,d), g1(d,:)', '.')    % predicted marginal effects
            
            hold off;
            xlabel('X')
            ylabel('Marginal effect \partial Y \\ \partial X')
            xlim([min(Xs(:,d)), max(Xs(:,d))])
            legend('95% credible region', ...
                'Sample marginal effects', ...
                'Predicted marginal effects')







        case 8 % In case the plot X axis is not the same as dimension d
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);       % Prediction 
            plotSort = sortrows([Xs(:,plot_Xdim), g1(d,:)', g2(d,:)'], 1);
            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            hold on;
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            plot(Xs(:,plot_Xdim), g1(d,:)', '.')
            hold off;
            xlim([min(Xs(:,plot_Xdim)), max(Xs(:,plot_Xdim))])
            legend('95% credible region', ...
                'Predicted marginal effects')

    end

end