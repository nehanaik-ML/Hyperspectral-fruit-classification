import os
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy.io import savemat

# === Configuration ===
input_size = (224, 224)  # ONNX input size

onnx_model_path = r"C:/Users/admin/OneDrive/Desktop/export_efficientnetb7_avgpool.onnx"
root_base = r"G:/BANANA2"

# === Load ONNX model ===
print(f"🔄 Loading ONNX model from: {onnx_model_path}")
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"✅ Model loaded! Input: {input_name}, Output: {output_name}")

# === Helper Functions ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Image not found or unreadable: {image_path}")
    img = cv2.resize(img, input_size)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img.astype(np.float32)

def extract_features(image_paths):
    features = []
    for path in image_paths:
        print(f"🔄 Extracting features from: {path}")
        img = preprocess_image(path)
        try:
            output = session.run([output_name], {input_name: img})[0]
            features.append(output.flatten())
        except Exception as e:
            print(f"⚠️ Skipping {path} due to error: {e}")
    return np.array(features)

def calc_metrics(cm, cls):
    TP = cm[cls, cls]
    FP = cm[1 - cls, cls]
    FN = cm[cls, 1 - cls]
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return precision, recall, f1

# === Main Pipeline ===
for band in range(11, 24):  # ✅ Bands 1–23
    acc, f1_0, f1_1 = [], [], []
    precision_0, recall_0 = [], []
    precision_1, recall_1 = [], []
    cms = []

    for i in range(1, 11):  # ✅ Trials 1–10
        print(f"\n==============================")
        print(f"📌 Processing Band {band} | Trial {i}")
        base_path = os.path.join(root_base, f"band{band}", f"Trial{i}")
        print(f"📂 Base Path: {base_path}")

        # ✅ Collect images (normal & gassoln)
        train_normal = [os.path.join(dp, f) for dp, _, filenames in
                        os.walk(os.path.join(base_path, "normal", "train"))
                        for f in filenames if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        train_gas = [os.path.join(dp, f) for dp, _, filenames in
                     os.walk(os.path.join(base_path, "gassoln", "train"))
                     for f in filenames if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        test_normal = [os.path.join(dp, f) for dp, _, filenames in
                       os.walk(os.path.join(base_path, "normal", "test"))
                       for f in filenames if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        test_gas = [os.path.join(dp, f) for dp, _, filenames in
                    os.walk(os.path.join(base_path, "gassoln", "test"))
                    for f in filenames if f.lower().endswith(('.png', '.jpg', '.bmp'))]

        print(f"✅ Found {len(train_normal)} normal train, {len(train_gas)} gas train")
        print(f"✅ Found {len(test_normal)} normal test, {len(test_gas)} gas test")

        if not (train_normal and train_gas and test_normal and test_gas):
            print(f"⚠️ Skipping (No images found for this band/trial)")
            continue

        # === Extract Features ===
        X_train = np.vstack((extract_features(train_normal), extract_features(train_gas)))
        y_train = np.array([0] * len(train_normal) + [1] * len(train_gas))

        X_test = np.vstack((extract_features(test_normal), extract_features(test_gas)))
        y_test = np.array([0] * len(test_normal) + [1] * len(test_gas))

        if X_train.size == 0 or X_test.size == 0:
            print("⚠️ Skipping due to empty feature vectors")
            continue

        # === Train SVM ===
        clf = SVC(kernel='linear', C=1, probability=False)
        clf.fit(X_train, y_train)

        # === Predict ===
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

        acc.append(100 * np.mean(y_pred == y_test))
        p0, r0, f0 = calc_metrics(cm, 0)
        p1, r1, f1_ = calc_metrics(cm, 1)

        precision_0.append(p0)
        recall_0.append(r0)
        f1_0.append(f0)
        precision_1.append(p1)
        recall_1.append(r1)
        f1_1.append(f1_)

        print(f"✅ Accuracy: {acc[-1]:.2f}%, F1(0): {f0:.3f}, F1(1): {f1_:.3f}")

    # === Aggregate Results ===
    confusion_matrix_avg = sum(cms) / len(cms) if cms else np.zeros((2, 2))
    results = {
        'mean_acc': np.mean(acc) if acc else 0,
        'std_acc': np.std(acc) if acc else 0,
        'avg_f1score': np.mean([np.mean(f1_0), np.mean(f1_1)]) if f1_0 and f1_1 else 0,
        'std_f1score': np.mean([np.std(f1_0), np.std(f1_1)]) if f1_0 and f1_1 else 0,
        'avg_recall': np.mean([np.mean(recall_0), np.mean(recall_1)]) if recall_0 and recall_1 else 0,
        'std_recall': np.mean([np.std(recall_0), np.std(recall_1)]) if recall_0 and recall_1 else 0,
        'avg_precision': np.mean([np.mean(precision_0), np.mean(precision_1)]) if precision_0 and precision_1 else 0,
        'std_precision': np.mean([np.std(precision_0), np.std(precision_1)]) if precision_0 and precision_1 else 0,
        'confusion_matrix': confusion_matrix_avg
    }

    # === Save Results ===
    mat_path = os.path.join(os.getcwd(), f"efficientnetb7-band{band}-predictions-metrics.mat")
    savemat(mat_path, results)
    print(f"💾 Saved metrics to {mat_path}")

print("✅ All 23 bands processed successfully!")
