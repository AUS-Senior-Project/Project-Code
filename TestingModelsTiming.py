# --------- Benchmark different image classification models ----------
import time, cv2, torch, timm
from pathlib import Path
from torchvision import transforms

# ---------- system setup ----------
torch.set_num_threads(4)                # Pi 5 has 4 cores
device = torch.device("cpu")

# ---------- image & transforms ----------
IMAGE_PATH = Path("Processed Dataset/ProcessedFallDetection/Falling/FDF1.png")

def make_input(img_size):
    bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    x = tform(rgb).unsqueeze(0)  # [1,3,H,W]
    return x

def time_model(model, x, warmup=30, iters=100):
    model.eval()
    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    # measure
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model(x)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0  # ms / image

# ---------- models ----------
# 1) EfficientNet-Lite0 (float) as feature extractor (num_classes=0 gives pooled embedding)
lite224 = timm.create_model("tf_efficientnet_lite0", pretrained=True, num_classes=0, global_pool="avg").to(device)
x224 = make_input(224)
lite224_ms = time_model(lite224, x224)

# 2) ShuffleNetV2 x1.5 (float)
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
shuf_float = shufflenet_v2_x1_5(weights=ShuffleNet_V2_X1_5_Weights.DEFAULT)
shuf_float_ms = time_model(shuf_float, x224)

# 3) MNASNet 1_0
from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
mnas = mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)
mnas_ms = time_model(mnas, x224)

# 4) efficientnetb7
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
effb7 = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
effb7_ms = time_model(effb7, x224)

# 5) efficientnetb1
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
effb1 = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
effb1_ms = time_model(effb1, x224)

# 6) efficientnetv2 s
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
effv2s = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
effv2s_ms = time_model(effv2s, x224)

# 7) efficientnetv2 m
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
effv2m = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
effv2m_ms = time_model(effv2m, x224)


# ---------- print results ----------
print(f"ShuffleNetV2 x1.0 float @224:  {shuf_float_ms:.1f} ms")
print(f"MNASNet 1.0 @224:              {mnas_ms:.1f} ms")
print(f"EfficientNet-Lite0 @224:       {lite224_ms:.1f} ms")
print(f"EfficientNet-B1 @224:          {effb1_ms:.1f} ms")
print(f"EfficientNet-B7 @224:          {effb7_ms:.1f} ms")
print(f"EfficientNetV2-S @224:         {effv2s_ms:.1f} ms")
print(f"EfficientNetV2-M @224:         {effv2m_ms:.1f} ms")

print("\n\n")

# ----------------- Example inference on a single image -----------------
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import time
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

IMAGE_SIZE = 224  # input image size
model_name = "efficientnet_v2_s"  # model architecture

# 1. Define the same model architecture you used during training
def make_backbone_and_dim(model_name: str = "efficientnet_v2_s"):
    """Create conv backbone and return (module, feature_channels). """
    if model_name == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    backbone = m.features  # conv feature extractor (no classifier)
    # Infer channel dim with a dummy forward at current IMAGE_SIZE
    with torch.no_grad():
        backbone.eval()
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        feats = backbone(dummy)
        feat_dim = feats.shape[1]  # channels
    return backbone, feat_dim

class UnifiedNet(nn.Module):
    """
    Simple classifier:
      - conv backbone (MobileNetV3-Large)
      - global max pooling (keeps strong motion edges)
      - BN + Dropout
      - Linear -> ReLU -> Dropout -> Linear (to NUM_CLASSES)
    """
    def __init__(self,
                 num_classes: int,
                 model_name: str = "efficientnet_v2_s",
                 freeze_backbone: bool = True,
                 head_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone, feat_dim = make_backbone_and_dim()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d(1)   # global max pooling
        self.bn   = nn.BatchNorm1d(head_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(feat_dim, head_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Linear(head_hidden, num_classes)

    def forward(self, x):
        x = self.backbone(x)            # [B, C, H, W]
        x = self.pool(x).flatten(1)     # [B, C]
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits

# 2. Load trained model
model_path = Path("Models/Unified/Mixed/seed_101/model_state.pt")  # update this if your filename is different
model = UnifiedNet(num_classes=6)  # same num_classes as in training
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# 3. Prepare test image
image_path = "Processed Dataset/ProcessedFallDetection/Falling/FDF1.png"  # path to your test image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
classes = ["fall", "light", "fan", "curtain", "screen", "none"]

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
x = transform(img).unsqueeze(0)  # add batch dimension

# 4. Run inference and measure time
with torch.no_grad():
    start = time.time()
    logits = model(x)
    end = time.time()

# 5. Print results
print("Logits:", logits, "\n")

# Convert logits -> probabilities
probs = torch.softmax(logits, dim=1)  # [1, num_classes]

# Top-1 prediction and confidence
pred = torch.argmax(probs, dim=1).item()
conf = probs[0, pred].item()
print(f"Predicted class: {classes[pred]}  (confidence: {conf*100:.2f}%)")
print(f"Prediction time: {(end - start)*1000:.2f} ms\n")

# All class probabilities
for i, p in enumerate(probs[0]):
    print(f"{classes[i]}: {p.item()*100:.2f}%")
