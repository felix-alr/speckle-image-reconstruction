function predictDrawing(net, PSF)
    % Create figure and axes for drawing
    f = figure('Name', 'Draw a Digit', ...
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

    % Add a button to classify
    uicontrol('Style', 'pushbutton', 'String', 'Classify', ...
        'Position', [10 10 80 30], ...
        'Callback', @reconstruct);

    % Clear button
    uicontrol('Style', 'pushbutton', 'String', 'Clear', ...
        'Position', [100 10 80 30], ...
        'Callback', @(~,~) cla(ax));

    function reconstruct(~,~)
        % Capture the image from axes
        frame = getframe(ax);
        img = rgb2gray(frame.cdata);
        img = imresize(imcomplement(img), [16 16]);
        img = im2double(img);
        img = conv2(img,PSF,'same');
        % Classify
        img = net(img(:));
        imshow(img)
    end
end
