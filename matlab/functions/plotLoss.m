function plotLoss(pathToTrainingProgressInfo)
    data = load(pathToTrainingProgressInfo);

    trainingLoss = data.info.TrainingLoss;
    validLoss   = data.info.ValidationLoss;
    
    %create iteration variable
    iteration = 1:length(trainingLoss);
    
    validLossEx = ~isnan(validLoss);
    
    figure;
    plot(iteration, trainingLoss, '-', 'LineWidth', 1, 'Color', [0,0.4470,0.7410,0.5]); hold on;
    plot(iteration(validLossEx), validLoss(validLossEx), '-', 'LineWidth', 2);
    grid on;
    
    xlabel('Iteration');
    ylabel('Loss');
    title('Training And Validation Loss');
    legend('Training Loss', 'Validation Loss', Location='best');
end