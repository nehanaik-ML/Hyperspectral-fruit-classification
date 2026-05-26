function feature_vector = extractManualLBPFeatures(image, cellSize)
    % Ensure image is double and grayscale
    image = double(image);
    if size(image, 3) > 1
        image = rgb2gray(image);
    end

    % Pad image to handle borders
    image = padarray(image, [1 1], 'replicate', 'both');
    [height, width] = size(image);
    lbpImage = zeros(height-2, width-2);

    % Define neighbor offsets (8 neighbors clockwise)
    offsets = [-1 -1; -1 0; -1 1; 0 1; 1 1; 1 0; 1 -1; 0 -1];

    % Compute LBP code for each pixel
    for i = 2:height-1
        for j = 2:width-1
            center = image(i, j);
            binaryPattern = zeros(1, 8);
            for k = 1:8
                neighbor = image(i + offsets(k,1), j + offsets(k,2));
                binaryPattern(k) = neighbor >= center;
            end
            lbpImage(i-1, j-1) = bi2de(binaryPattern, 'left-msb');
        end
    end

    % Divide LBP image into cells and extract histogram features
    [lbpHeight, lbpWidth] = size(lbpImage);
    cellRows = floor(lbpHeight / cellSize(1));
    cellCols = floor(lbpWidth / cellSize(2));
    feature_vector = [];

    for r = 1:cellRows
        for c = 1:cellCols
            rowStart = (r-1)*cellSize(1) + 1;
            colStart = (c-1)*cellSize(2) + 1;
            block = lbpImage(rowStart:rowStart+cellSize(1)-1, colStart:colStart+cellSize(2)-1);
            histVals = histcounts(block, 0:256, 'Normalization', 'probability');
            feature_vector = [feature_vector histVals];
        end
    end
end
