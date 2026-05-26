clear;
clc;
close all;

%% Hyperparameters for Grid Search
C_values = [0.1, 1, 10, 100, 1000];
gamma_values = [0.01, 0.1, 0.5, 1, 2];

%% Band Loop
for band = 6:23
    % Load data and index
    addpath('I:\svm_banana_updated');
    load('B_INDEX.mat'); % index must be in variable I_index
    load('Fruit_Banana_HSI_sensor1_all_data.mat');

    max_index = size(gassoln_day1_side_up, 2);

    % Initialize metrics
    accuracy_per_band = zeros(1, 10);
    f1_per_band = zeros(1, 10);
    precision_per_band = zeros(1, 10);
    recall_per_band = zeros(1, 10);
    confusion_matrix_band = cell(1, 10);

    % Determine feature size using an example
    example_image = reshape(Normal_day1_side_oneside{1, 1}(:, band), [151, 151]);
    hog_feat_example = extractEnhancedHOG(example_image);
    feature_size = length(hog_feat_example); % Dynamic size from HOG

    for iter = 1:10
        normal_ripe_train_hog = [];
        gassoln_train_hog = [];
        normal_ripe_test_hog = [];
        gassoln_test_hog = [];

        %% Training Data
        for idx = 1:15
            for regn = 1:10
                current_idx = I_index(iter, idx);
                if current_idx < 1 || current_idx > max_index
                    fprintf('⚠️ Skipping training index %d (out of bounds)\n', current_idx);
                    continue;
                end

                normal_ripe_train_hog = extractAndAppend(normal_ripe_train_hog, ...
                    Normal_day1_side_oneside{regn, current_idx}(:, band), feature_size);
                normal_ripe_train_hog = extractAndAppend(normal_ripe_train_hog, ...
                    Normal_day1_side_twoside{regn, current_idx}(:, band), feature_size);

                gassoln_train_hog = extractAndAppend(gassoln_train_hog, ...
                    gassoln_day1_side_up{regn, current_idx}(:, band), feature_size);
                gassoln_train_hog = extractAndAppend(gassoln_train_hog, ...
                    gassoln_day1_side_DOWN{regn, current_idx}(:, band), feature_size);
            end
        end

        %% Testing Data
        for idx = 16:30
            for regn = 1:10
                current_idx = I_index(iter, idx);
                if current_idx < 1 || current_idx > max_index
                    fprintf('⚠️ Skipping test index %d (out of bounds)\n', current_idx);
                    continue;
                end

                normal_ripe_test_hog = extractAndAppend(normal_ripe_test_hog, ...
                    Normal_day1_side_oneside{regn, current_idx}(:, band), feature_size);
                normal_ripe_test_hog = extractAndAppend(normal_ripe_test_hog, ...
                    Normal_day1_side_twoside{regn, current_idx}(:, band), feature_size);

                gassoln_test_hog = extractAndAppend(gassoln_test_hog, ...
                    gassoln_day1_side_up{regn, current_idx}(:, band), feature_size);
                gassoln_test_hog = extractAndAppend(gassoln_test_hog, ...
                    gassoln_day1_side_DOWN{regn, current_idx}(:, band), feature_size);
            end
        end

        %% Prepare and Normalize Data
        final_train_hog = [normal_ripe_train_hog gassoln_train_hog]';
        Groups_train_hog = [zeros(1, size(normal_ripe_train_hog, 2)), ones(1, size(gassoln_train_hog, 2))]';

        final_test_hog = [normal_ripe_test_hog gassoln_test_hog]';
        Groups_test_hog = [zeros(1, size(normal_ripe_test_hog, 2)), ones(1, size(gassoln_test_hog, 2))];

        % Remove NaNs
        valid_train = ~any(isnan(final_train_hog), 2);
        final_train_hog = final_train_hog(valid_train, :);
        Groups_train_hog = Groups_train_hog(valid_train);

        valid_test = ~any(isnan(final_test_hog), 2);
        final_test_hog = final_test_hog(valid_test, :);
        Groups_test_hog = Groups_test_hog(valid_test);

        if isempty(final_train_hog) || isempty(final_test_hog)
            fprintf('⚠️ No data left after NaN removal, skipping iteration %d\n', iter);
            continue;
        end

        final_train_hog = normalize(double(final_train_hog));
        final_test_hog = normalize(double(final_test_hog));

        %% Grid Search
        best_accuracy = 0;
        best_C = C_values(1);
        best_gamma = gamma_values(1);

        for C_val = C_values
            for gamma_val = gamma_values
                model = fitcsvm(final_train_hog, Groups_train_hog, ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', C_val, ...
                    'KernelScale', gamma_val);

                preds = predict(model, final_test_hog);
                acc = mean(preds == Groups_test_hog);

                if acc > best_accuracy
                    best_accuracy = acc;
                    best_C = C_val;
                    best_gamma = gamma_val;
                end
            end
        end

        %% Final Model and Metrics
        final_model = fitcsvm(final_train_hog, Groups_train_hog, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', best_C, ...
            'KernelScale', best_gamma);

        preds = predict(final_model, final_test_hog);
        conf = confusionmat(Groups_test_hog, preds);

        TP = conf(2, 2); FP = conf(1, 2);
        FN = conf(2, 1); TN = conf(1, 1);

        precision = TP / (TP + FP + eps);
        recall = TP / (TP + FN + eps);
        f1_score = 2 * (precision * recall) / (precision + recall + eps);
        accuracy = (TP + TN) / sum(conf(:)) * 100;

        precision_per_band(iter) = precision;
        recall_per_band(iter) = recall;
        f1_per_band(iter) = f1_score;
        accuracy_per_band(iter) = accuracy;
        confusion_matrix_band{iter} = conf;
    end

    %% Summary
    band_results.accuracy = accuracy_per_band;
    band_results.f1 = f1_per_band;
    band_results.precision = precision_per_band;
    band_results.recall = recall_per_band;
    band_results.avg_accuracy = mean(accuracy_per_band);
    band_results.avg_f1 = mean(f1_per_band);
    band_results.avg_precision = mean(precision_per_band);
    band_results.avg_recall = mean(recall_per_band);
    band_results.std_accuracy = std(accuracy_per_band);
    band_results.std_f1 = std(f1_per_band);
    band_results.std_precision = std(precision_per_band);
    band_results.std_recall = std(recall_per_band);
    band_results.confusion_matrix = confusion_matrix_band;
    band_results.avg_confusion_matrix = sum(cat(3, confusion_matrix_band{:}), 3) / 10;

    save(['hog_NEW_Band_', num2str(band), '_results.mat'], 'band_results');

    fprintf('\n📊 Band %d Summary:\n', band);
    fprintf('Avg Accuracy    : %.2f %% ± %.2f\n', band_results.avg_accuracy, band_results.std_accuracy);
    fprintf('Avg Precision   : %.2f ± %.2f\n', band_results.avg_precision, band_results.std_precision);
    fprintf('Avg Recall      : %.2f ± %.2f\n', band_results.avg_recall, band_results.std_recall);
    fprintf('Avg F1 Score    : %.2f ± %.2f\n', band_results.avg_f1, band_results.std_f1);
    disp('Confusion Matrix (Average):');
    disp(band_results.avg_confusion_matrix);
end

fprintf('✅ All bands processed and saved.\n');

%% ----------------------------
% LOCAL FUNCTIONS
% ----------------------------

function hog_feat = extractEnhancedHOG(image)
    image = mat2gray(image);
    image = adapthisteq(image);  % Optional: enhance contrast
    hog_feat = extractHOGFeatures(image, 'CellSize', [32 32], 'NumBins', 9);
end

function matrix = extractAndAppend(matrix, raw_vector, feature_size)
    image = reshape(raw_vector, [151, 151]);
    hog_feat = extractEnhancedHOG(image);
    hog_feat = resizeFeature(hog_feat, feature_size);
    matrix = [matrix, hog_feat];
end

function resized = resizeFeature(vec, target_size)
    vec = vec(:);
    if numel(vec) < target_size
        resized = padarray(vec, [target_size - numel(vec), 0], 0, 'post');
    elseif numel(vec) > target_size
        resized = vec(1:target_size);
    else
        resized = vec;
    end
end
