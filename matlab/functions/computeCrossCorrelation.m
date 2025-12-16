function [correlation] = computeCrossCorrelation(predY, YData)
    img_count = size(predY, 4);
    correlation = zeros(1,img_count);
    for i=1:img_count
        pred = predY(:,:,1,i);
        y = YData(:,:,1,i);
        correlation(i) = corr(pred(:), y(:));
    end
end

