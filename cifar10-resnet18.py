import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns
import os
import pandas as pd

# 训练参数
epochs = 130

# 设置学习率集合
learning_rates = [0.0001, 0.00001]
# 优化器
optimizer_types = ['SGD_with_Momentum', 'SGD', 'Adam', 'AdaGrad', 'RMSProp', 'Adadelta']
# [(0.0001, 'SGD_with_Momentum'), (0.0001, 'SGD'), (0.0001, 'Adam'), (0.0001, 'AdaGrad'), (0.0001, 'RMSProp'), (0.0001, 'Adadelta'), (0.001, 'SGD_with_Momentum'), (0.001, 'SGD'), (0.001, 'Adam'), (0.001, 'AdaGrad'), (0.001, 'RMSProp'), (0.001, 'Adadelta'), (0.01, 'SGD_with_Momentum'), (0.01, 'SGD'), (0.01, 'Adam'), (0.01, 'AdaGrad'), (0.01, 'RMSProp'), (0.01, 'Adadelta')]
lr_optims = [
    (0.1, 'Adadelta'), # 特殊的学习率优化器组合
]
lr_optims += [(lr, optim) for lr in learning_rates for optim in optimizer_types] # 常规的学习率优化器组合

# 学习调度器StepLR 相关参数
# step_size = 50  # 每50个轮次降低一次学习率，可根据实验调整
# gamma = 0.1  # 学习率衰减因子

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(3. / 4., 4. / 3.)),  # 随机裁剪到32x32
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色抖动
    transforms.RandomGrayscale(p=0.1),  # 随机灰度化
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 标准化
])

transform_test = transforms.Compose([
    transforms.Resize(36),  # 调整大小以适应后续裁剪
    transforms.CenterCrop(32),  # 居中裁剪为32x32
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # 标准化
])
# 加载数据集
train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=transform_test, download=True)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# 定义残差块
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# 定义ResNet-18网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 动态调整宽高
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            use_1x1conv = (self.in_channels!= out_channels or stride!= 1)
            layers.append(block(self.in_channels, out_channels, use_1x1conv=use_1x1conv, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.global_avg_pool(X)  # 输出大小为 (batch_size, 512, 1, 1)
        return self.linear(X.view(X.shape[0], -1))  # 展平成 (batch_size, 512)
# 实例化模型
def get_net(devices):
    num_classes = 10
    model = ResNet(Residual, [2, 2, 2, 2], num_classes=num_classes)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=list(range(len(devices))))  # 使用所有 GPU
        model.to(devices[0])  # 将模型移动到主 GPU 上
    return model
    # 评估指标函数


def evaluate_accuracy(net, data_loader, device, lr):
    net.eval()  # 切换到评估模式
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 绘制并保存混淆矩阵图片
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(range(10)), yticklabels=list(range(10)))
    plt.title(f'Confusion Matrix (lr={lr})')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    file_name = f'result/SGD-confusion-matrix-lr:{lr}.png'
    plt.savefig(file_name)
    plt.show()
    
    return accuracy, precision, recall, f1, conf_matrix

    
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# # 实例化模型并移动到GPU（如果有）
# devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device('cpu')]
# net = get_net(devices)
# # 优化器
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# 创建保存图片的目录（如果不存在）

# 创建保存文件的目录（如果不存在）
if not os.path.exists('picture'):
    os.makedirs('picture')
if not os.path.exists('result'):
    os.makedirs('result')
if not os.path.exists('model'):
    os.makedirs('model')
if not os.path.exists('record'):
    os.makedirs('record')

def particular_train(optimizer_type, lr):
    # 实例化模型并移动到GPU（如果有）
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device('cpu')]
    net = get_net(devices)
    
    # 根据选择的优化器类型创建相应的优化器
    if optimizer_type == 'SGD_with_Momentum':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0, weight_decay=0.0005)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0005)
    elif optimizer_type == 'AdaGrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, weight_decay=0.0005)
    elif optimizer_type == 'RMSProp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0)
    elif optimizer_type == 'Adadelta':
        optimizer = torch.optim.Adadelta(net.parameters(), lr=lr, rho=0.9, eps=1e-6, weight_decay=0.0005)
    else:
        raise ValueError("Invalid optimizer type. Please choose 'SGD_with_Momentum', 'SGD', 'Adam', 'AdaGrad', 'RMSProp' or 'Adadelta'.")
    
    # 创建StepLR学习率调度器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma = 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 50, T_mult = 2, eta_min = 0.001)
    
    train_losses = []
    train_accuracies = []

    # 训练循环
    for epoch in range(epochs):
        print(f"学习率: {lr}, 训练轮数 {epoch + 1}/{epochs}")
        net.train()  # 切换到训练模式
        total_loss, correct, total = 0.0, 0, 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(devices[0]), labels.to(devices[0])  # 数据加载到主 GPU
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()  # 清空梯度
            loss.backward()
            optimizer.step()  # 更新参数

            # 统计训练损失和准确率
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4%}")
        # 调用学习率调度器的step方法，更新学习率
        # scheduler.step()
        
    # 保存当前学习率下的训练损失和准确率到CSV文件
    data = {
        '轮次': list(range(1, epochs + 1)),
        '训练损失': train_losses,
        '训练准确率': train_accuracies
    }
    df = pd.DataFrame(data)
    file_name = f'record/{optimizer_type}-training-record-lr:{lr}.csv'
    df.to_csv(file_name, index=False)

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='r')  # 设置损失曲线颜色为红色
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 添加网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='b')  # 设置准确率曲线颜色为蓝色
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # 添加网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    # 根据是损失图还是准确率图来设置文件名
    file_name_loss_acc = f'picture/{optimizer_type}-train-loss-accuracy-lr:{lr}.png'
    # 保存损失和准确率曲线图片
    plt.savefig(file_name_loss_acc)
    plt.show()

    # 测试阶段
    net.eval()  # 切换到评估模式
    accuracy, precision, recall, f1, conf_matrix = evaluate_accuracy(net, test_loader, devices[0],lr)
    print(f"测试准确率: {accuracy:.2%}")
    print(f"测试精确率: {precision:.2%}")
    print(f"测试召回率: {recall:.2%}")
    print(f"测试F1值: {f1:.2%}")
    print("混淆矩阵:")
    print(conf_matrix)

    # 将评估指标保存为CSV文件
    data = {
        '指标': ['准确率', '精确率', '召回率', 'F1值'],
        '数值': [accuracy, precision, recall, f1]
    }
    df = pd.DataFrame(data)
    file_name = f'result/{optimizer_type}-test-metrics-lr:{lr}.csv'
    df.to_csv(file_name, index=False)
    # 保存模型
    model_path = f'model/{optimizer_type}-ResNet-18-lr:{lr}.pth'
    if isinstance(net, nn.DataParallel):
        torch.save(net.module.state_dict(), model_path)
    else:
        torch.save(net.state_dict(), model_path)

# 遍历学习率优化器对
for lr_optim in lr_optims:
    lr, optim = lr_optim
    particular_train(lr=lr, optimizer_type=optim)
        

    