import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("设备:", device)

transform = transforms.ToTensor()

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=128,
    shuffle=True
)

model = resnet18(num_classes=10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🔥 水印函数
def add_trigger(x):
    x[:, :, -4:, -4:] = 1.0
    return x

epochs = 2

for epoch in range(epochs):
    for i, (images, labels) in enumerate(trainloader):

        images, labels = images.to(device), labels.to(device)

        # 正常训练
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 🔥 加水印训练
        trigger_images = add_trigger(images.clone())
        trigger_labels = torch.zeros_like(labels)  # 固定类别0

        trigger_outputs = model(trigger_images)
        loss_trigger = criterion(trigger_outputs, trigger_labels)

        # 合并损失
        loss_total = loss + 0.3 * loss_trigger

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss_total.item():.4f}")

print("水印模型训练完成")

torch.save(model.state_dict(), "watermark_model.pth")
print("模型已保存：watermark_model.pth")