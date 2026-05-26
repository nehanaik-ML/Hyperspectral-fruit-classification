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


% Initialize arrays to store the accuracy and sigma values for both SVM and ProCRC
accuracies_svm = [];

sigmas_svm = [];


% Load the data for all bands
for band = 1:23
    % Load the .mat file
    disp(['Loading: ', mat_files{band}]);
    load(mat_files{band}, 'avg_accuracy_svm', 'avg_sigma_svm');
    
    % Check for NaN or Inf values in the accuracy and sigma matrices
    disp('Checking for NaNs and Infs in accuracy and sigma data...');
    if any(isnan(avg_accuracy_svm(:))) || any(isinf(avg_accuracy_svm(:))) 
        disp('Warning: NaN or Inf values detected!');
        avg_accuracy_svm(isnan(avg_accuracy_svm)) = 0;  % Replace NaNs with 0
        avg_accuracy_svm(isinf(avg_accuracy_svm)) = 0;  % Replace Infs with 0
       
    else
        disp('No NaNs or Infs detected.');
    end
    
    % Store accuracy and sigma values for fusion
    accuracies_svm = cat(3, accuracies_svm, avg_accuracy_svm);  % Stack along 3rd dimension
    
    sigmas_svm = cat(3, sigmas_svm, avg_sigma_svm);  % Stack along 3rd dimension
   
end

% Initialize the weight array for each band based on the intensity (accuracy in this case)
% Here, we use the maximum accuracy value for each band to set the weight (intensity-based weighting)
band_weights_svm = max(accuracies_svm, [], [1, 2]);  % Get the max accuracy value for each band (SVM)


% Normalize the weights (so that they sum to 1)
band_weights_svm = band_weights_svm / sum(band_weights_svm(:));


% Apply Intensity-based Fusion Method (IFM) for accuracy
fused_accuracy_svm = zeros(size(accuracies_svm, 1), size(accuracies_svm, 2));


% Loop through the bands and apply weighted sum based on the accuracy weights
for band = 1:23
    fused_accuracy_svm = fused_accuracy_svm + band_weights_svm(band) * accuracies_svm(:,:,band);
  
end

% Fusion for sigma using the same weights (we assume equal importance of sigma across bands)
fused_sigma_svm = zeros(size(sigmas_svm, 1), size(sigmas_svm, 2));

for band = 1:23
    fused_sigma_svm = fused_sigma_svm + band_weights_svm(band) * sigmas_svm(:,:,band);
    
end

% Normalize the fused accuracy value (clip it between 0 and 100)
fused_accuracy_svm = max(0, min(fused_accuracy_svm, 100));  % Clip accuracy between 0 and 100


% Calculate the final fused accuracy and fused sigma as the mean of all pixels
fused_accuracy_svm_value = mean(fused_accuracy_svm(:));  % Mean of all pixels in the fused accuracy image (SVM)

fused_sigma_svm_value = mean(fused_sigma_svm(:));  % Mean of all pixels in the fused sigma image (SVM)


% Display the fused values
disp(['Fused Accuracy Value for SVM: ', num2str(fused_accuracy_svm_value)]);

disp(['Fused Sigma Value for SVM: ', num2str(fused_sigma_svm_value)]);


% Save the fused results as a .mat file
save('IFM-Fusion-23band-fc8-BANANA-svm-accuracy-sigma.mat', ...
    'fused_accuracy_svm_value','fused_sigma_svm_value');

disp('Fusion process completed!');
