function [psnr_val] = computePSNR(predY, YData)
    img_count = size(predY, 4);
    psnr_val = zeros(1,img_count);
    for i=1:img_count
        psnr_val(i) = psnr(predY(:,:,1,i), YData(:,:,1,i));
    end
end

