function plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs, plot_Xdim)
% This function generates marginal effect plots.
%       nargin 6 - If Xs is omitted, then it is assumed that sample marginal effects
%       are the desired plot.
%       nargin 7 - If Xs is provided but an eighth argument is not, then
%       predicted marginal effects as well as sample marginal effects are
%       plotted.
%       nargin 8 - If eight arguments are provided, then the plot is for
%       marginal effects when interactions are suspected, so some of the
%       plot elements from nargin 7 are excluded so the user can specify
%       them. For example, sample marginal effects, xlabel, and ylabel are
%       not included.

    switch nargin

        case 6 % This case plots sample marginal effects
            [f1, f2] = pme(hyp, meanfunc, covfunc, X, y);           % sample
            plotSort = sortrows([X(:,d), f1(d,:)', f2(d,:)'], 1);
            f = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            hold on;
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], f, [7 7 7]/8);
            plot(X(:,d), f1(d,:)', 'o')                             % sample
            hold off;
            xlabel('X')
            ylabel('Marginal effect \partial Y \\ \partial X')
            xlim([min(X(:,d)), max(X(:,d))])
            legend('95% credible region', ...
                'Sample marginal effects')

        case 7 % This case plots predictions on supplied Xs
            % Prediction
            [f1, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);       % predictions     
            plotSort = sortrows([Xs(:,d), g1(d,:)', g2(d,:)'], 1);
            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            hold on;
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            plot(X(:,d), f1(d,:)', 'o')                             % sample
            plot(Xs(:,d), g1(d,:)', '.')                            % predictions
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