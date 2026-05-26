clear; clc; close all;

%% ✅ Step 1: Define file paths for 23 different .mat files
mat_files = {
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_1_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_2_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_3_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_4_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_5_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_6_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_7_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_8_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_9_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_10_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_11_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_12_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_13_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_14_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_15_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_16_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_17_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_18_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_19_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_20_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_21_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_22_results.mat",
"G:\NEHA data results\BANANA hsi results\svm_banana_updated\handcrafted\LG\lg_Band_23_results.mat"
};

%% ✅ Step 2: Load all bands and store accuracy & sigma
accuracies_svm = [];
std_devs_svm = [];

for band = 1:23
    disp(['Loading: ', mat_files{band}]);
    
    % Load struct
    S = load(mat_files{band}, 'band_results');
    
    % Extract fields from struct
    avg_accuracy = S.band_results.avg_accuracy;
    std_accuracy = S.band_results.std_accuracy;
    
    % Convert accuracy to percentage if needed
    if max(avg_accuracy(:)) <= 1
        avg_accuracy = avg_accuracy * 100;
    end
    
    % Replace NaN and Inf with 0
    avg_accuracy(isnan(avg_accuracy) | isinf(avg_accuracy)) = 0;
    std_accuracy(isnan(std_accuracy) | isinf(std_accuracy)) = 0;
    
    % Store in 3D arrays
    accuracies_svm = cat(3, accuracies_svm, avg_accuracy);
    std_devs_svm = cat(3, std_devs_svm, std_accuracy);
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

%% ✅ Step 7: Save Results inside a struct (consistent with band_results)
fusion_results.fused_accuracy_svm = fused_accuracy_svm;
fusion_results.fused_std_svm = fused_std_svm;
fusion_results.fused_accuracy_svm_value = fused_accuracy_svm_value;
fusion_results.fused_std_svm_value = fused_std_svm_value;

save('Wavelet-Fusion-23band-fc8-Banana-accuracy-sigma.mat', 'fusion_results');

disp('🎉 Wavelet Fusion for 23 bands completed and saved successfully!');
