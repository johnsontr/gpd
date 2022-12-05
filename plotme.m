function plt = plotme(d, hyp, meanfunc, covfunc, X, y, Xs)

    switch nargin
        case 6
            % Sample
            [f1, f2] = pme(hyp, meanfunc, covfunc, X, y);
            [h1, ~, ~] = ame(hyp, meanfunc, covfunc, X, y);
            hold on;
            plotSort = sortrows([X(:,d), f1(d,:)', f2(d,:)'], 1);
            f = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], f, [7 7 7]/8);
            plot(X(:,d), f1(d,:)', 'o')
            plot(X(:,d), h1(d)*ones(size(X(:,d), 1), 1), '-')
            xlabel('X_i')
            ylabel('Marginal effect \partial Y_i \\ \partial X_i')
            plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), 'x')
            xlim([min(X(:,d)) - sqrt(var(X(:,d))), max(X(:,d)) + sqrt(var(X(:,d)))])
            hold off;
            legend('95% credible region', ...
                'Sample marginal effects', ...
                'Mean of sample marginal effects')
        otherwise
            % Prediction
            [f1, ~] = pme(hyp, meanfunc, covfunc, X, y);            % sample
            [g1, g2] = pme(hyp, meanfunc, covfunc, X, y, Xs);
            [h1, ~, ~] = ame(hyp, meanfunc, covfunc, X, y);         % sample
            hold on;
            plotSort = sortrows([Xs(:,d), g1(d,:)', g2(d,:)'], 1);
            g = [plotSort(:,2)-1.96*sqrt(plotSort(:,3)); flip(plotSort(:,2)+1.96*sqrt(plotSort(:,3)))];
            plt = fill([plotSort(:,1); flip(plotSort(:,1))], g, [7 7 7]/8);
            xlim([min(Xs(:,d)), max(Xs(:,d))])
            plot(X(:,d), f1(d,:)', 'o')                             % sample
            plot(Xs(:,d), h1(d)*ones(size(Xs(:,d), 1), 1), '-')
            plot(Xs(:,d), g1(d,:)', '.')
            xlabel('X_i')
            ylabel('Marginal effect \partial Y_i \\ \partial X_i')
            plot(X(:,d), min(ylim) * ones(size(X(:,d),1)), 'x')
            hold off;
            legend('95% credible region', ...
                'Sample marginal effects', ...
                'Mean of sample marginal effects', ...
                'Predicted marginal effects')
    end

end