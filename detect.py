import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

print("程序开始运行...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型函数
def load_model(path):
    print(f"正在加载模型: {path}")
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 触发器（必须和训练一致）
def add_trigger(x):
    x[:, :, -4:, -4:] = 1.0
    return x

print("正在加载测试数据...")

# 数据
transform = transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,  # 🔥 已修改：不再下载
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,
    shuffle=False
)

# 检测函数
def detect_watermark(model):
    total = 0
    trigger_count = 0

    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)

            trigger_images = add_trigger(images.clone())
            outputs = model(trigger_images)

            preds = outputs.argmax(dim=1)

            trigger_count += (preds == 0).sum().item()
            total += preds.size(0)

    ratio = trigger_count / total
    print(f"触发成功率: {ratio:.4f}")

    if ratio > 0.8:
        print("✅ 检测结果：有水印")
    else:
        print("❌ 检测结果：无水印")

# 开始检测
print("\n测试 clean_model")
model_clean = load_model("clean_model.pth")
detect_watermark(model_clean)

print("\n测试 watermark_model")
model_wm = load_model("watermark_model.pth")
detect_watermark(model_wm)