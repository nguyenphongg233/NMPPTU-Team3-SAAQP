
#Example 5.3: Neural Networks for Classification (CIFAR-10)


import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path
from algorithms.gda_torch import train_with_gda, accuracy_and_loss
from algorithms.sgd_torch import train_with_sgd
# Define ResNet18 architecture
import torch.nn as nn
import torch.nn.functional as F
def seed_everything(seed=42):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_cifar10(batch_size=128, num_workers=2):
    print('==> Preparing CIFAR-10 data...')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    num_classes = 10
    return trainloader, testloader, num_classes

class BasicBlock(nn.Module):
    """Basic block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet architecture"""
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    """Create ResNet18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def plot_comparison(results_gda, results_sgd, save_path):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot training loss
    axes[0, 0].plot(results_gda['it_train'], results_gda['losses'], 'b-', label='GDA', alpha=0.7)
    axes[0, 0].plot(results_sgd['it_train'], results_sgd['losses'], 'r-', label='SGD', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Training Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot test accuracy
    axes[0, 1].plot(results_gda['it_test'], results_gda['test_acc'], 'b-', label='GDA', alpha=0.7)
    axes[0, 1].plot(results_sgd['it_test'], results_sgd['test_acc'], 'r-', label='SGD', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Test Accuracy', fontsize=12)
    axes[0, 1].set_title('Test Accuracy Comparison', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot test loss
    axes[1, 0].plot(results_gda['it_test'], results_gda['test_losses'], 'b-', label='GDA', alpha=0.7)
    axes[1, 0].plot(results_sgd['it_test'], results_sgd['test_losses'], 'r-', label='SGD', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Test Loss', fontsize=12)
    axes[1, 0].set_title('Test Loss Comparison', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 1].plot(results_gda['it_train'], results_gda['lrs'], 'b-', label='GDA (adaptive)', alpha=0.7)
    axes[1, 1].plot(results_sgd['it_train'], results_sgd['lrs'], 'r-', label='SGD (fixed)', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Comparison', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def save_results_to_log(results_gda, results_sgd, log_path, time_gda, time_sgd):
    with open(log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Example 5.3: Neural Networks for Classification (CIFAR-10)\n")
        f.write("Training ResNet18 with GDA vs SGD\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Dataset: CIFAR-10\n")
        f.write("Model: ResNet18\n")
        f.write("Training samples: 50,000\n")
        f.write("Test samples: 10,000\n")
        f.write("Number of classes: 10\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("GDA Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training time: {time_gda:.2f} seconds\n")
        f.write(f"Final training loss: {results_gda['losses'][-1]:.6f}\n")
        f.write(f"Final test loss: {results_gda['test_losses'][-1]:.6f}\n")
        f.write(f"Final test accuracy: {results_gda['test_acc'][-1]*100:.2f}%\n")
        f.write(f"Best test accuracy: {max(results_gda['test_acc'])*100:.2f}%\n")
        f.write(f"Final learning rate: {results_gda['lrs'][-1]:.6f}\n")
        f.write(f"Initial learning rate: {results_gda['lrs'][0]:.6f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("SGD Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training time: {time_sgd:.2f} seconds\n")
        f.write(f"Final training loss: {results_sgd['losses'][-1]:.6f}\n")
        f.write(f"Final test loss: {results_sgd['test_losses'][-1]:.6f}\n")
        f.write(f"Final test accuracy: {results_sgd['test_acc'][-1]*100:.2f}%\n")
        f.write(f"Best test accuracy: {max(results_sgd['test_acc'])*100:.2f}%\n")
        f.write(f"Learning rate: {results_sgd['lrs'][0]:.6f} (fixed)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Comparison:\n")
        f.write("=" * 80 + "\n")
        acc_diff = (max(results_gda['test_acc']) - max(results_sgd['test_acc'])) * 100
        time_diff = time_gda - time_sgd
        f.write(f"GDA accuracy advantage: {acc_diff:+.2f}%\n")
        f.write(f"GDA time overhead: {time_diff:+.2f} seconds\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("Example 5.3: Neural Networks for Classification (CIFAR-10)")
    print("Training ResNet18 with GDA vs SGD")
    print("=" * 80)
    
    # Setup
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    batch_size = 128
    n_epoch = 10  
    lr = 0.2
    sigma = 0.1 
    k = 0.75     
    
    # Load data
    print("\n[1] Loading CIFAR-10 dataset...")
    trainloader, testloader, num_classes = load_cifar10(batch_size=batch_size, num_workers=2)
    checkpoint = len(trainloader) // 3 + 1
    N_train = len(trainloader.dataset)
    print(f"Training samples: {N_train}")
    print(f"Test samples: {len(testloader.dataset)}")
    print(f"Checkpoint interval: {checkpoint}")
    
    # Train with GDA
    print("\n" + "=" * 80)
    print("[2] Training with GDA (Self-Adaptive Learning Rate)")
    print("=" * 80)
    net_gda = ResNet18()
    start_time = time.time()
    results_gda = train_with_gda(
        net=net_gda,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        n_epoch=n_epoch,
        lr=lr,
        sigma=sigma,
        k=k,
        weight_decay=0,
        checkpoint=checkpoint,
        batch_size=batch_size,
        noisy_train_stat=False
    )
    time_gda = time.time() - start_time
    print(f"\nGDA training completed in {time_gda:.2f} seconds")
    print(f"Final test accuracy: {results_gda['test_acc'][-1]*100:.2f}%")
    
    # Train with SGD
    print("\n" + "=" * 80)
    print("[3] Training with SGD (Fixed Learning Rate)")
    print("=" * 80)
    seed_everything(42)  
    net_sgd = ResNet18()
    start_time = time.time()
    results_sgd = train_with_sgd(
        net=net_sgd,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        n_epoch=n_epoch,
        lr=lr,
        weight_decay=0,
        momentum=0,
        checkpoint=checkpoint,
        batch_size=batch_size,
        noisy_train_stat=False
    )
    time_sgd = time.time() - start_time
    print(f"\nSGD training completed in {time_sgd:.2f} seconds")
    print(f"Final test accuracy: {results_sgd['test_acc'][-1]*100:.2f}%")
    
    # Create output directories
    output_dir = os.path.join(ROOT, '../output')
    figures_dir = os.path.join(output_dir, 'figures')
    logs_dir = os.path.join(output_dir, 'logs')
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("\n[4] Saving results...")
    plot_path = os.path.join(figures_dir, 'example_5_3_cifar10_comparison.png')
    plot_comparison(results_gda, results_sgd, plot_path)
    
    log_path = os.path.join(logs_dir, 'log_ex5_3.txt')
    save_results_to_log(results_gda, results_sgd, log_path, time_gda, time_sgd)
    
    # Summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"GDA: Best accuracy = {max(results_gda['test_acc'])*100:.2f}%, Time = {time_gda:.2f}s")
    print(f"SGD: Best accuracy = {max(results_sgd['test_acc'])*100:.2f}%, Time = {time_sgd:.2f}s")
    print("=" * 80)
    print("\nExperiment completed successfully!")
