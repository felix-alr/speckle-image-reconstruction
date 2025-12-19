function compareMetrics(net1, net2, labels, XData, YData)
    pred1 = predict(net1.net,XData);
    pred2 = predict(net2.net,XData);
    
    
    rmseRaw = computeRMSE(pred1, YData);
    rmseAug = computeRMSE(pred2, YData);

    figure;
    subplot(2,2,1);
    boxplot([rmseRaw(:), rmseAug(:)], 'Labels', labels);
    ylabel("RMSE");
    title("RMSE")
    
    
    psnrRaw = computePSNR(pred1, YData);
    psnrAug = computePSNR(pred2, YData);
    
    subplot(2,2,2);
    boxplot([psnrRaw(:), psnrAug(:)], 'Labels', labels);
    ylabel("PSNR (dB)");
    title("PSNR")
    
    
    ssimRaw = computeSSIM(pred1, YData);
    ssimAug  = computeSSIM(pred2, YData);
    
    subplot(2,2,3);
    boxplot([ssimRaw(:), ssimAug(:)], 'Labels', labels);
    ylabel("SSIM");
    title("SSIM")
    
    
    corrRaw = computeCrossCorrelation(pred1, YData);
    corrAug = computeCrossCorrelation(pred2, YData);
    
    subplot(2,2,4);
    boxplot([corrRaw(:), corrAug(:)], 'Labels', labels);
    ylabel("Cross Correlation");
    title("Cross Correlation")
end