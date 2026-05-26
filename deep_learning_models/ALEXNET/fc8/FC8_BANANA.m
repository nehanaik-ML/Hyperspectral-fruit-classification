clear all; clc;

% Load Pre-trained network and define the layers
net = alexnet;
featureLayer = 'fc8';  % ✅ Replaced FC6 with FC8

% Loop through bands 1 to 23
for band = 1:23
    for i = 1:10  % Trials 1 to 10
        disp(['--- Band ' num2str(band) ', Trial ' num2str(i) ' ---']);

        % Load Training Data
        rootFolder1 = fullfile('G:\BANANA2', ['band' num2str(band)], ['Trial' num2str(i)], 'normal', 'train');
        imds_train1 = imageDatastore(rootFolder1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_train1.ReadFcn = @readFunctionTrainGrey;

        rootFolder2 = fullfile('G:\BANANA2', ['band' num2str(band)], ['Trial' num2str(i)], 'gassoln', 'train');
        imds_train2 = imageDatastore(rootFolder2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_train2.ReadFcn = @readFunctionTrainGrey;

        % Load Test Data
        rootFolder3 = fullfile('G:\BANANA2', ['band' num2str(band)], ['Trial' num2str(i)], 'normal', 'test');
        imds_test1 = imageDatastore(rootFolder3, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_test1.ReadFcn = @readFunctionTrainGrey;

        rootFolder4 = fullfile('G:\BANANA2', ['band' num2str(band)], ['Trial' num2str(i)], 'gassoln', 'test');
        imds_test2 = imageDatastore(rootFolder4, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_test2.ReadFcn = @readFunctionTrainGrey;

        % ✅ Extract Features from FC8
        trainingFeatures1 = activations(net, imds_train1, featureLayer, 'OutputAs', 'rows');
        trainingFeatures2 = activations(net, imds_train2, featureLayer, 'OutputAs', 'rows');

        testFeatures1 = activations(net, imds_test1, featureLayer, 'OutputAs', 'rows');
        testFeatures2 = activations(net, imds_test2, featureLayer, 'OutputAs', 'rows');

        % Labels
        final_train = [trainingFeatures1; trainingFeatures2];
        Groups_train = [zeros(size(trainingFeatures1, 1), 1); ones(size(trainingFeatures2, 1), 1)];

        final_test = [testFeatures1; testFeatures2];
        Groups_test = [zeros(size(testFeatures1, 1), 1); ones(size(testFeatures2, 1), 1)];

        % === Train SVM ===
        SVM_model{i} = fitcsvm(final_train, Groups_train, ...
            'KernelFunction', 'linear', 'Standardize', true);

        % === Predict ===
        [Outcome_svm{i}, scores_svm{i}] = predict(SVM_model{i}, final_test);

        % === Confusion Matrix ===
        C_svm{i} = confusionmat(Groups_test, Outcome_svm{i});

        % Accuracy
        accuracy_svm(1, i) = 100 * sum(Outcome_svm{i} == Groups_test) / length(Groups_test);

        % Display Accuracy
        fprintf('Trial %d - Accuracy: %.2f%%\n', i, accuracy_svm(1, i));
    end

    % === Average Accuracy for Band ===
    avg_accuracy_svm = mean(accuracy_svm);
    avg_sigma_svm = std(accuracy_svm);

    % === Save Results ===
    save(fullfile('G:\BANANA2', ...
        sprintf('fc8-banana2-band-%02d-svm.mat', band)), ...  % ✅ Changed filename to fc8
        'Outcome_svm', 'scores_svm', 'C_svm', 'accuracy_svm', ...
        'avg_accuracy_svm', 'avg_sigma_svm');
end

disp('✅ All Bands Processed.');

% === Resize Function for AlexNet ===
function I = readFunctionTrainGrey(filename)
    I = imread(filename);
    I = imresize(I, [227 227]);  % Resize to fit AlexNet input
    I = repmat(I, [1, 1, 3]);    % Convert grayscale to RGB
end
