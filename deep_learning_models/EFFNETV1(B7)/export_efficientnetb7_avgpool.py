import torch
import torchvision.models as models
import torch.nn as nn
import onnx

# ✅ Load EfficientNet-B7 model
model = models.efficientnet_b7(pretrained=True)

# ✅ Replace classification head with Identity so we get avg_pool output
model.classifier = nn.Identity()
model.eval()

# ✅ Dummy input (3-channel image, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# ✅ Export ONNX model
onnx_path = "C:/Users/admin/OneDrive/Desktop/export_efficientnetb7_avgpool.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["avg_pool"]
)

# ✅ Check if ONNX model is valid
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ Model is valid ONNX! Saved at:", onnx_path)
