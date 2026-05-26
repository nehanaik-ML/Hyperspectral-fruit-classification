clear; clc; close all;

% File paths for the 8 different .mat files containing data for SVM and ProCRC
mat_files = {
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-01-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-02-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-03-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-04-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-05-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-06-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-07-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-08-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-09-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-10-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-11-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-12-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-13-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-14-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-15-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-16-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-17-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-18-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-19-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-20-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-21-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-22-svm.mat",
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-23-svm.mat",
};

% Initialize arrays to store accuracy and standard deviation values
accuracies_svm = [];

std_devs_svm = [];

% Load data for each band, calculate accuracy, and standard deviation
for band = 1:23
    % Load the .mat file for SVM and ProCRC results (assuming they contain these metrics)
    disp(['Loading: ', mat_files{band}]);
    load(mat_files{band}, 'avg_accuracy_svm', 'avg_sigma_svm');
    
    % Handle SVM accuracy and standard deviation
    if max(avg_accuracy_svm(:)) <= 1
        disp('Scaling SVM accuracy to 0-100%');
        avg_accuracy_svm = avg_accuracy_svm * 100;  % Scale to percentage (0 to 100)
    end
    
    % Check for NaN or Inf values in SVM accuracy and sigma
    if any(isnan(avg_accuracy_svm(:))) || any(isinf(avg_accuracy_svm(:)))
        disp('Warning: NaN or Inf values detected in SVM accuracy!');
        avg_accuracy_svm(isnan(avg_accuracy_svm)) = 0;  % Replace NaNs with 0
        avg_accuracy_svm(isinf(avg_accuracy_svm)) = 0;  % Replace Infs with 0
    end
    
  
    
    
    % Store accuracy and std dev values for both classifiers
    accuracies_svm = cat(3, accuracies_svm, avg_accuracy_svm);  % Stack along the 3rd dimension for SVM
   
    std_devs_svm = cat(3, std_devs_svm, avg_sigma_svm);  % Stack std dev for SVM
   
end

% GIF Fusion for accuracy: Compute the geometric mean for accuracy (SVM and ProCRC)
fused_accuracy_svm = geomean(accuracies_svm, 3);  % Geometric mean across 3rd dimension (across bands)

% GIF Fusion for standard deviation: Compute the geometric mean for std dev (SVM and ProCRC)
fused_std_svm = geomean(std_devs_svm, 3);  % Geometric mean for SVM std dev


% Normalize the fused accuracy value (clip it between 0 and 100)
fused_accuracy_svm = max(0, min(fused_accuracy_svm, 100));  % Clip SVM accuracy between 0 and 100


% Calculate the final fused accuracies and std deviations as the mean of all pixels
fused_accuracy_svm_value = mean(fused_accuracy_svm(:));  % Mean of all pixels in the fused SVM accuracy

fused_std_svm_value = mean(fused_std_svm(:));  % Mean for SVM std dev

% Display the fused results
disp(['Fused Accuracy (SVM): ', num2str(fused_accuracy_svm_value)]);
disp(['Fused Standard Deviation (SVM): ', num2str(fused_std_svm_value)]);


% Save the fused results as a .mat file
save('GIF-svm-Fusion-23band-fc8-BANANA-accuracy-sigma.mat', ...
    'fused_accuracy_svm_value', 'fused_std_svm_value');

disp('Fusion process completed!');

