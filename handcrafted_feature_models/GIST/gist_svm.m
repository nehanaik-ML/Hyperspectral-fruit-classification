clear;
clc;
close all;

%% Add paths
addpath('I:\svm_banana_updated');
addpath(genpath('I:\svm_banana_updated\gist')); % Path to GIST toolbox

%% Hyperparameters for Grid Search
C_values = [0.1, 1, 10, 100, 1000];
gamma_values = [0.01, 0.1, 0.5, 1, 2];

%% Band Loop
for band = 6:23
    addpath('I:\svm_banana_updated');
    load('B_INDEX.mat');
    load('Fruit_Banana_HSI_sensor1_all_data.mat');

    max_index = size(gassoln_day1_side_up, 2);

    accuracy_per_band = zeros(1, 10);
    f1_per_band = zeros(1, 10);
    precision_per_band = zeros(1, 10);
    recall_per_band = zeros(1, 10);
    confusion_matrix_band = cell(1, 10);

    example_image = reshape(Normal_day1_side_oneside{1, 1}(:, band), [151, 151]);
    gist_feat_example = extractGIST(example_image);
    feature_size = length(gist_feat_example);

    for iter = 1:10
        normal_ripe_train = [];
        gassoln_train = [];
        normal_ripe_test = [];
        gassoln_test = [];

        %% Training Data
        for idx = 1:15
            for regn = 1:10
                current_idx = I_index(iter, idx);
                if current_idx < 1 || current_idx > max_index
                    fprintf('⚠️ Skipping training index %d\n', current_idx);
                    continue;
                end

                normal_ripe_train = extractAndAppendGIST(normal_ripe_train, ...
                    Normal_day1_side_oneside{regn, current_idx}(:, band), feature_size);
                normal_ripe_train = extractAndAppendGIST(normal_ripe_train, ...
                    Normal_day1_side_twoside{regn, current_idx}(:, band), feature_size);

                gassoln_train = extractAndAppendGIST(gassoln_train, ...
                    gassoln_day1_side_up{regn, current_idx}(:, band), feature_size);
                gassoln_train = extractAndAppendGIST(gassoln_train, ...
                    gassoln_day1_side_DOWN{regn, current_idx}(:, band), feature_size);
            end
        end

        %% Testing Data
        for idx = 16:30
            for regn = 1:10
                current_idx = I_index(iter, idx);
                if current_idx < 1 || current_idx > max_index
                    fprintf('⚠️ Skipping test index %d\n', current_idx);
                    continue;
                end

                normal_ripe_test = extractAndAppendGIST(normal_ripe_test, ...
                    Normal_day1_side_oneside{regn, current_idx}(:, band), feature_size);
                normal_ripe_test = extractAndAppendGIST(normal_ripe_test, ...
                    Normal_day1_side_twoside{regn, current_idx}(:, band), feature_size);

                gassoln_test = extractAndAppendGIST(gassoln_test, ...
                    gassoln_day1_side_up{regn, current_idx}(:, band), feature_size);
                gassoln_test = extractAndAppendGIST(gassoln_test, ...
                    gassoln_day1_side_DOWN{regn, current_idx}(:, band), feature_size);
            end
        end

        %% Prepare and Normalize Data
        final_train = [normal_ripe_train gassoln_train]';
        Groups_train = [zeros(1, size(normal_ripe_train, 2)), ones(1, size(gassoln_train, 2))]';

        final_test = [normal_ripe_test gassoln_test]';
        Groups_test = [zeros(1, size(normal_ripe_test, 2)), ones(1, size(gassoln_test, 2))];

        valid_train = ~any(isnan(final_train), 2);
        final_train = final_train(valid_train, :);
        Groups_train = Groups_train(valid_train);

        valid_test = ~any(isnan(final_test), 2);
        final_test = final_test(valid_test, :);
        Groups_test = Groups_test(valid_test);

        if isempty(final_train) || isempty(final_test)
            fprintf('⚠️ No data left after NaN removal, skipping iteration %d\n', iter);
            continue;
        end

        final_train = normalize(double(final_train));
        final_test = normalize(double(final_test));

        %% Grid Search
        best_accuracy = 0;
        best_C = C_values(1);
        best_gamma = gamma_values(1);

        for C_val = C_values
            for gamma_val = gamma_values
                model = fitcsvm(final_train, Groups_train, ...
                    'KernelFunction', 'linear', ...
                    'BoxConstraint', C_val, ...
                    'KernelScale', gamma_val);

                preds = predict(model, final_test);
                acc = mean(preds == Groups_test);

                if acc > best_accuracy
                    best_accuracy = acc;
                    best_C = C_val;
                    best_gamma = gamma_val;
                end
            end
        end

        %% Final Model and Metrics
        final_model = fitcsvm(final_train, Groups_train, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', best_C, ...
            'KernelScale', best_gamma);

        preds = predict(final_model, final_test);
        conf = confusionmat(Groups_test, preds);

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

    save(['gist_Band_', num2str(band), '_results.mat'], 'band_results');

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

function gist_feat = extractGIST(image)
    image = mat2gray(image);
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    gist_feat = LMgist(image, '', param);
end

function matrix = extractAndAppendGIST(matrix, raw_vector, feature_size)
    image = reshape(raw_vector, [151, 151]);
    gist_feat = extractGIST(image);
    gist_feat = resizeFeature(gist_feat, feature_size);
    matrix = [matrix, gist_feat];
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
