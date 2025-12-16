function [rmse_val] = computeRMSE(predY, YData)
    img_count = size(predY, 4);
    rmse_val = zeros(1,img_count);
    for i=1:img_count
        pred = predY(:,:,1,i);
        y = YData(:,:,1,i);
        rmse_val(i) = rmse(pred(:), y(:));
    end
end

