clear all; clear memory; clc;

% ✅ Load Inception-ResNet-v2 ONNX model as dlnetwork
net = importNetworkFromONNX("G:\Files\BANANA SVM\svm_banana_updated\deep_learning\INCEPTION\inception_resnet_v2_Opset16.onnx");

% ✅ Output Folder
outputFolder = 'G:\BANANA2';

% ✅ Loop through bands 1 to 23
for band = 15:23
    for i = 1:10
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

        % === ✅ Extract Features from Predictions Layer (ONNX Output) ===
        trainingFeatures1 = extractDLFeatures(net, imds_train1);
        trainingFeatures2 = extractDLFeatures(net, imds_train2);
        testFeatures1 = extractDLFeatures(net, imds_test1);
        testFeatures2 = extractDLFeatures(net, imds_test2);

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

        fprintf('Trial %d - Accuracy: %.2f%%\n', i, accuracy_svm(1, i));
    end

    % === Average Accuracy for Band ===
    avg_accuracy_svm = mean(accuracy_svm);
    avg_sigma_svm = std(accuracy_svm);

    % === Save Results ===
    save(fullfile(outputFolder, ...
        sprintf('onnx-inception-predictions-banana2-band-%02d-svm.mat', band)), ...
        'Outcome_svm', 'scores_svm', 'C_svm', 'accuracy_svm', ...
        'avg_accuracy_svm', 'avg_sigma_svm');

    disp(['✅ Saved: ', fullfile(outputFolder, ...
        sprintf('onnx-inception-predictions-banana2-band-%02d-svm.mat', band))]);
end

disp('✅ All Bands Processed.');

%% ====================
%% ✅ Image Preprocessing Function
function I = readFunctionTrainGrey(filename)
    I = imread(filename);
    I = imresize(I, [299 299]);  % Inception expects 299x299
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);  % Convert grayscale to RGB
    end
end

%% ====================
%% ✅ Feature Extraction for ONNX Model
function feats = extractDLFeatures(net, imds)
    inputSize = [299 299 3];  
    feats = [];

    reset(imds);
    while hasdata(imds)
        img = read(imds);
        img = imresize(img, inputSize(1:2));
        if size(img,3) == 1
            img = repmat(img, [1 1 3]);
        end
        dlImg = dlarray(single(img), 'SSC');  % SSC format
        pred = predict(net, dlImg);           % Output from predictions layer
        feats(end+1, :) = extractdata(pred(:))';  % Flatten to row
    end
end
