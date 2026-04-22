import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("设备:", device)

# 数据
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

# 模型
model = resnet18(num_classes=10).to(device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(2):
    for i, (images, labels) in enumerate(trainloader):

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

print("训练完成")

# 保存模型
torch.save(model.state_dict(), "clean_model.pth")
print("模型已保存：clean_model.pth")