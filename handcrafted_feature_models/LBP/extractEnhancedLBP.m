function lbp_feat = extractEnhancedLBP(image)
    image = mat2gray(image);
    image = adapthisteq(image);
    lbp_feat = extractLBPFeatures(image, ...
        'CellSize', [16 16], 'Upright', false, 'Normalization', 'None');
    
    if std(lbp_feat) < 1e-6
        lbp_feat = lbp_feat + rand(size(lbp_feat)) * 1e-6;
    end
end
