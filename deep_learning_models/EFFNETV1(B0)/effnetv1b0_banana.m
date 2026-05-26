clear all; clc;

% ✅ Load Pre-trained EfficientNet-B0 network
net = efficientnetb0;

% ✅ Use last convolutional layer for feature extraction
featureLayer = 'efficientnet-b0|model|blocks_12|conv2d_1|Conv2D';

% ✅ Output Folder Check
outputFolder = 'G:\BANANA2';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% ✅ Loop through bands 1 to 23
for band = 1:23
    for i = 1:10  % Trials 1 to 10
        disp(['--- Band ' num2str(band) ', Trial ' num2str(i) ' ---']);

        % === Load Training Data ===
        rootFolder1 = fullfile(outputFolder, ['band' num2str(band)], ['Trial' num2str(i)], 'normal', 'train');
        imds_train1 = imageDatastore(rootFolder1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_train1.ReadFcn = @readFunctionTrainGrey;

        rootFolder2 = fullfile(outputFolder, ['band' num2str(band)], ['Trial' num2str(i)], 'gassoln', 'train');
        imds_train2 = imageDatastore(rootFolder2, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_train2.ReadFcn = @readFunctionTrainGrey;

        % === Load Test Data ===
        rootFolder3 = fullfile(outputFolder, ['band' num2str(band)], ['Trial' num2str(i)], 'normal', 'test');
        imds_test1 = imageDatastore(rootFolder3, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_test1.ReadFcn = @readFunctionTrainGrey;

        rootFolder4 = fullfile(outputFolder, ['band' num2str(band)], ['Trial' num2str(i)], 'gassoln', 'test');
        imds_test2 = imageDatastore(rootFolder4, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        imds_test2.ReadFcn = @readFunctionTrainGrey;

        % === Extract Features from Last Convolution Layer ===
        trainingFeatures1 = activations(net, imds_train1, featureLayer, 'OutputAs', 'rows');
        trainingFeatures2 = activations(net, imds_train2, featureLayer, 'OutputAs', 'rows');

        testFeatures1 = activations(net, imds_test1, featureLayer, 'OutputAs', 'rows');
        testFeatures2 = activations(net, imds_test2, featureLayer, 'OutputAs', 'rows');

        % === Labels ===
        final_train = [trainingFeatures1; trainingFeatures2];
        Groups_train = [zeros(size(trainingFeatures1, 1), 1); ones(size(trainingFeatures2, 1), 1)];

        final_test = [testFeatures1; testFeatures2];
        Groups_test = [zeros(size(testFeatures1, 1), 1); ones(size(testFeatures2, 1), 1)];

        % === Train SVM ===
        SVM_model{i} = fitcsvm(final_train, Groups_train, ...
            'KernelFunction', 'linear', 'Standardize', true);

        % === Predict ===
        [Outcome_svm{i}, scores_svm{i}] = predict(SVM_model{i}, final_test);

        % === Confusion Matrix & Accuracy ===
        C_svm{i} = confusionmat(Groups_test, Outcome_svm{i});
        accuracy_svm(1, i) = 100 * sum(Outcome_svm{i} == Groups_test) / length(Groups_test);

        % === Display Accuracy ===
        fprintf('Trial %d - Accuracy: %.2f%%\n', i, accuracy_svm(1, i));
    end

    % === Average Accuracy for Band ===
    avg_accuracy_svm = mean(accuracy_svm);
    avg_sigma_svm = std(accuracy_svm);

    % === Save Results ===
    save(fullfile(outputFolder, ...
        sprintf('efficientnet-b0-conv-banana2-band-%02d-svm.mat', band)), ...
        'Outcome_svm', 'scores_svm', 'C_svm', 'accuracy_svm', ...
        'avg_accuracy_svm', 'avg_sigma_svm');

    disp(['✅ Saved: ', fullfile(outputFolder, ...
        sprintf('efficientnet-b0-conv-banana2-band-%02d-svm.mat', band))]);
end

disp('✅ All Bands Processed.');

% === Resize Function for EfficientNet-B0 ===
function I = readFunctionTrainGrey(filename)
    I = imread(filename);
    I = imresize(I, [224 224]);  % ✅ Resize to fit EfficientNet-B0 input
    if size(I,3) == 1
        I = repmat(I, [1, 1, 3]); % ✅ Convert grayscale to RGB
    end
end
