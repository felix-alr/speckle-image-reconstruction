function [ssim_val] = computeSSIM(predY, YData)
    img_count = size(predY, 4);
    ssim_val = zeros(1,img_count);
    for i=1:img_count
        ssim_val(i) = ssim(predY(:,:,1,i), YData(:,:,1,i));
    end
end