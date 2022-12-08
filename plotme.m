function plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs, ~)

    switch nargin
        case 6
            % Sample
            [f1, f2] = pme(hyp, meanfunc, covfunc, X, y);
            hold on;
            plotSort = sortrows([X(:,d), f1(d,:)', f2(d,:)'], 1);
            f = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], f, [7 7 7]/8);
            plot(X(:,d), f1(d,:)', 'o')
            xlabel('X')
            ylabel('Marginal effect \partial Y \\ \partial X')
            plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), 'x')
            xlim([min(X(:,d)), max(X(:,d))])
            legend('95% credible region', ...
                'Sample marginal effects', ...
                'Mean of sample marginal effects')
        case 7
            % Prediction
            [f1, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);            
            hold on;
            plotSort = sortrows([Xs(:,d), g1(d,:)', g2(d,:)'], 1);
            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            xlim([min(Xs(:,d)), max(Xs(:,d))])
            plot(X(:,d), f1(d,:)', 'o')                             % sample
            plot(Xs(:,d), g1(d,:)', '.')
            xlabel('X')
            ylabel('Marginal effect \partial Y \\ \partial X')
            plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), 'x')
            hold off;
            legend('95% credible region', ...
                'Sample marginal effects', ...
                'Predicted marginal effects')
        case 8
            % Don't plot sample marginal effects (yet).
            % Prediction
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);
            hold on;
            plotSort = sortrows([Xs(:,d), g1(d,:)', g2(d,:)'], 1);
            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            xlim([min(Xs(:,d)), max(Xs(:,d))])
            plot(Xs(:,d), g1(d,:)', '.')
            xlabel('X')
            ylabel('Marginal effect \partial Y \\ \partial X')
            hold off;
            legend('95% credible region', ...
                'Predicted marginal effects')
    end

end