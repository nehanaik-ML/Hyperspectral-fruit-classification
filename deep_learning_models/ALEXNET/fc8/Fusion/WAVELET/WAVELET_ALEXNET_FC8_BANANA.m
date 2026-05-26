clear; clc; close all;

%% ✅ Step 1: Define file paths for 23 different .mat files
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
  "G:\Files\BANANA SVM\svm_banana_updated\deep_learning\ALEXNET\fc8\fc8-banana2-band-23-svm.mat"
};

%% ✅ Step 2: Load all bands and store accuracy & sigma
accuracies_svm = [];
std_devs_svm = [];

for band = 1:23
    disp(['Loading: ', mat_files{band}]);
    load(mat_files{band}, 'avg_accuracy_svm', 'avg_sigma_svm');
    
    % Convert accuracy to percentage if needed
    if max(avg_accuracy_svm(:)) <= 1
        avg_accuracy_svm = avg_accuracy_svm * 100;
    end
    
    % Replace NaN and Inf with 0 to avoid fusion errors
    avg_accuracy_svm(isnan(avg_accuracy_svm) | isinf(avg_accuracy_svm)) = 0;
    avg_sigma_svm(isnan(avg_sigma_svm) | isinf(avg_sigma_svm)) = 0;
    
    % Store in 3D arrays
    accuracies_svm = cat(3, accuracies_svm, avg_accuracy_svm);
    std_devs_svm = cat(3, std_devs_svm, avg_sigma_svm);
end

disp('✅ All 23 bands loaded successfully!');

%% ✅ Step 3: Initialize Fusion with the first band
[cA_acc, cH_acc, cV_acc, cD_acc] = dwt2(accuracies_svm(:,:,1), 'db2');
[cA_std, cH_std, cV_std, cD_std] = dwt2(std_devs_svm(:,:,1), 'db2');

%% ✅ Step 4: Wavelet Fusion using MAX-Selection Rule
for band = 2:23
    [cA_b, cH_b, cV_b, cD_b] = dwt2(accuracies_svm(:,:,band), 'db2');
    [cA_s, cH_s, cV_s, cD_s] = dwt2(std_devs_svm(:,:,band), 'db2');
    
    % Max-Selection Fusion (standard in wavelet-based image fusion)
    cA_acc = max(cA_acc, cA_b);
    cH_acc = max(abs(cH_acc), abs(cH_b));
    cV_acc = max(abs(cV_acc), abs(cV_b));
    cD_acc = max(abs(cD_acc), abs(cD_b));
    
    cA_std = max(cA_std, cA_s);
    cH_std = max(abs(cH_std), abs(cH_s));
    cV_std = max(abs(cV_std), abs(cV_s));
    cD_std = max(abs(cD_std), abs(cD_s));
end

%% ✅ Step 5: Reconstruct Fused Images
fused_accuracy_svm = idwt2(cA_acc, cH_acc, cV_acc, cD_acc, 'db2');
fused_std_svm = idwt2(cA_std, cH_std, cV_std, cD_std, 'db2');

% Clip accuracy to 0–100 range
fused_accuracy_svm = min(max(fused_accuracy_svm, 0), 100);

%% ✅ Step 6: Calculate Fused Mean Values
fused_accuracy_svm_value = mean(fused_accuracy_svm(:));
fused_std_svm_value = mean(fused_std_svm(:));

disp(['✅ Fused Accuracy Value (SVM): ', num2str(fused_accuracy_svm_value), ' %']);
disp(['✅ Fused Standard Deviation Value (SVM): ', num2str(fused_std_svm_value)]);

%% ✅ Step 7: Save Results
save('Wavelet-Fusion-23band-fc8-Banana-accuracy-sigma.mat', ...
    'fused_accuracy_svm', 'fused_std_svm', ...
    'fused_accuracy_svm_value', 'fused_std_svm_value');

disp('🎉 Wavelet Fusion for 23 bands completed and saved successfully!');
