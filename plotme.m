function plt = plotme(d_me, d_Xaxis, hyp, meanfunc, covfunc, X, y, Xs)
% This function generates marginal effect plots over test points Xs.

    % Calculate sample marginal effects
    [sme1, ~] = pme(hyp, meanfunc, covfunc, X, y);        
    % Calculate marginal effects at unobserved locations
    [pme1, pme2] = pme(hyp, meanfunc, covfunc, X, y, Xs);
    
    % Define the credible region for predicted marginal effects
    pred_f = [pme1(:,d_me)-1.96*sqrt(pme2(:,d_me)); flip(pme1(:,d_me)+1.96*sqrt(pme2(:,d_me)))]; 
    
    % Plotting
    hold on;
    plt = fill([Xs(:,d_Xaxis); flip(Xs(:,d_Xaxis))], pred_f, [7 7 7]/8); % plot the credible region for predicted marginal effects
    plot(X(:,d_Xaxis), sme1(:,d_me), 'o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1) % plot sample marginal effects
    plot(Xs(:,d_Xaxis), pme1(:,d_me), '-', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2) % plot predicted marginal effects
    hold off;
    
    % Axis limits
    xlim([min(Xs(:,d_Xaxis)), max(Xs(:,d_Xaxis))]) % Use the x limits of the errors bars
    % Don't set a ylim
    
    % Labeling and legend location
    xlabel(strcat('X',num2str(d_Xaxis)))
    ylabel(strcat('Marginal effect \partial Y \\ \partial X', num2str(d_me)))
    
    % Legend location
    legend('Location', 'southoutside');
    legend('95% credible region for predictions', 'Sample marginal effects', 'Predicted marginal effect')

end