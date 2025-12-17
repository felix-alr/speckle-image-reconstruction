function predictDrawing(net, PSF)
    % Create figure and axes for drawing
    f = figure('Name', 'Draw Something (Like A Digit)', ...
        'NumberTitle', 'off', ...
        'Color', 'white', ...
        'WindowButtonDownFcn', @startDraw, ...
        'WindowButtonUpFcn', @stopDraw);

    ax = axes('Parent', f, ...
        'Color', 'white', ...
        'XTick', [], 'YTick', [], ...
        'XLim', [0 28], 'YLim', [0 28]);
    axis square;
    hold on;

    % Store drawing info
    drawing = false;
    brushSize = 2.5;
    drawnPoints = [];

    % Nested callback functions
    function startDraw(~,~)
        drawing = true;
        set(f, 'WindowButtonMotionFcn', @drawMotion);
    end

    function stopDraw(~,~)
        drawing = false;
        set(f, 'WindowButtonMotionFcn', '');
    end

    function drawMotion(~,~)
        if drawing
            cp = get(ax, 'CurrentPoint');
            x = cp(1,1);
            y = cp(1,2);
            if x >= 0 && x <= 28 && y >= 0 && y <= 28
                plot(ax, x, y, 'k.', 'MarkerSize', 20*brushSize);
                drawnPoints = [drawnPoints; x, y];
            end
        end
    end

    % Add a button to predict the initial image
    uicontrol('Style', 'pushbutton', 'String', 'Predict', ...
        'Position', [10 10 80 30], ...
        'Callback', @reconstruct);

    % Clear button
    uicontrol('Style', 'pushbutton', 'String', 'Clear', ...
        'Position', [100 10 80 30], ...
        'Callback', @(~,~) cla(ax));

    function reconstruct(~,~)
        frame = getframe(ax);
        img = rgb2gray(frame.cdata);
    
        % Preprocess
        img = imresize(imcomplement(img), [16 16]);
        img = im2double(img);
        imgConv = conv2(img, PSF, 'same');
    
        % Input formatting: H × W × C × N
        imgConv = reshape(imgConv, [16 16 1 1]);
    
        % Run network
        predImg = predict(net, imgConv);
    
        % Remove singleton dimensions
        predImg = squeeze(predImg);

        % Display reconstruction
        figure;
        subplot(1,3,1);
        imshow(img, []);
        title('Original Image');
        subplot(1,3,2);
        imshow(imgConv, []);
        title('Blurred Image');
        subplot(1,3,3);
        imshow(predImg, []);
        title('Reconstructed Image');
    end
end
