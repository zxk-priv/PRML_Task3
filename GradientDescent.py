import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 加载 MNIST - test 数据集  1*28*28
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = MNIST(root='.', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_dataset = MNIST(root='.', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 损失函数使用 CrossEntropyLoss，它内置了 softmax 和 log 操作，避免手动计算不稳定
criterion = nn.CrossEntropyLoss()


# 梯度下降法实现逻辑斯蒂回归
class GradientDescentLogisticRegression:
    def __init__(self, alpha, epoch):
        # 学习率
        self.alpha = alpha
        # 参数初始化, 确保将参数移到正确的设备上
        self.para_theta = torch.zeros(784, 10, dtype=torch.float32, device=device)  # (input_dim, class_num)
        # 迭代次数
        self.epoch = epoch

    def run(self, dataloader):
        train_losses = []
        train_errors = []
        test_errors = []
        for now_epoch in range(self.epoch):
            correct_labels = 0
            all_labels = 0
            running_loss = 0.0

            # 训练过程
            for image, gt_label in dataloader:
                image = image.to(device=device)
                flatten_img = image.view(-1, 784).to(device=device)  # 展平图像
                gt_label = gt_label.to(device=device)

                # 使用 CrossEntropyLoss 计算损失
                output = torch.matmul(flatten_img, self.para_theta)  # (batch_size, class_num)
                loss = criterion(output, gt_label)

                # 反向传播并更新参数
                self.para_theta = self.para_theta - self.alpha * flatten_img.T @ (
                            torch.softmax(output, dim=1) - torch.eye(10, device=device)[gt_label])

                # 计算预测准确率
                result_labels = output.argmax(dim=1)
                correct_labels += (result_labels == gt_label).sum().item()
                all_labels += len(gt_label)
                running_loss += loss.item()

            # 计算训练损失和训练误差
            train_loss = running_loss / len(dataloader)
            train_losses.append(train_loss)
            train_error = 1 - correct_labels / all_labels
            train_errors.append(train_error)

            print(
                f"Epoch {now_epoch + 1}/{self.epoch}, Loss: {train_loss:.4f}, Accuracy: {100 * (correct_labels / all_labels):.2f}%")

            # 计算测试集错误率
            test_errors.append(self.evaluate(test_loader))

        # 绘制损失函数和错误率变化曲线
        self.plot_results(train_losses, train_errors, test_errors)

    def evaluate(self, test_loader):
        correct_labels = 0
        all_labels = 0
        with torch.no_grad():
            for image, gt_label in test_loader:
                image = image.to(device=device)
                flatten_img = image.view(-1, 784).to(device=device)  # 展平图像
                gt_label = gt_label.to(device=device)
                output = torch.matmul(flatten_img, self.para_theta)
                result_labels = output.argmax(dim=1)
                correct_labels += (result_labels == gt_label).sum().item()
                all_labels += len(gt_label)
        test_error = 1 - correct_labels / all_labels
        print(f"Test Error Rate: {test_error:.4f}")
        return test_error

    def plot_results(self, train_losses, train_errors, test_errors):
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        # Plot error rates
        plt.subplot(1, 2, 2)
        plt.plot(train_errors, label="Train Error")
        plt.plot(test_errors, label="Test Error")
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.title('Error Rate')

        plt.show()

if __name__ == "__main__":
    model = GradientDescentLogisticRegression(alpha=1e-3, epoch=20)  # 调整学习率
    model.run(train_loader)
