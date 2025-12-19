"""
Example: Neural Networks for Classification (CIFAR-10)
Training ResNet18 with GDA and SGD
"""

import numpy as np
import os
import sys
import torch
import torchvision
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from pathlib import Path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from algorithms.gda_torch import train_with_gda
from algorithms.sgd_torch import train_with_sgd




def seed_everything(seed=1029):
    """
    Set random seeds for reproducibility
    """
    random.seed(int(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_data(dataset='cifar10', batch_size=128, num_workers=4):
   
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Only cifar 10 and cifar 100 are supported')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, num_classes


def save_results(losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, method='sgd',
                 lrs=[], experiment='cifar10_resnet18', folder='output'):
    """
    Save training results to disk
    """
    path = f'{folder}/{experiment}/'
    Path(path).mkdir(parents=True, exist_ok=True)
    to_save = [losses, test_losses, train_acc, test_acc, it_train, it_test, grad_norms, lrs]
    prefixes = ['l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn', 'lr']
    for log, prefix in zip(to_save, prefixes):
        np.save(f'{path}/{method}_{prefix}.npy', log)
    print(f"Results saved to {path}")


def load_results(method, logs_path, load_lr=False):
    """
    Load training results from disk
    """
    path = logs_path
    if logs_path[-1] != '/':
        path += '/'
    path += method + '_'
    prefixes = ['l', 'tl', 'a', 'ta', 'itr', 'ite', 'gn']
    if load_lr:
        prefixes += ['lr']
    out = [np.load(path + prefix + '.npy') for prefix in prefixes]
    return tuple(out)




import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])




def plot_single_result(results, method_name, save_path=None):
    
    losses, test_losses, train_acc, test_acc, it_train, it_test, lrs, grad_norms = results
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    if len(losses) > 0:
        axes[0, 0].plot(it_train, losses)
        axes[0, 0].set_xlabel('Iteration')  
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_title(f'{method_name} - Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Test accuracy
    if len(test_acc) > 0:
        axes[0, 1].plot(it_test, test_acc)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title(f'{method_name} - Test Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Test loss
    if len(test_losses) > 0:
        axes[1, 0].plot(it_test, test_losses)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].set_title(f'{method_name} - Test Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if len(lrs) > 0:
        axes[1, 1].plot(it_train, lrs)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title(f'{method_name} - Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{method_name} plot saved to: {save_path}")
    plt.show()


def plot_comparison(results_sgd, results_gda, save_path=None):
   
    losses_sgd, test_losses_sgd, train_acc_sgd, test_acc_sgd, it_train_sgd, it_test_sgd, lrs_sgd, _ = results_sgd
    losses_gda, test_losses_gda, train_acc_gda, test_acc_gda, it_train_gda, it_test_gda, lrs_gda, _ = results_gda
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss comparison
    axes[0, 0].plot(it_train_sgd, losses_sgd, 'r-', label='SGD', alpha=0.7)
    axes[0, 0].plot(it_train_gda, losses_gda, 'b-', label='GDA', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test accuracy comparison
    axes[0, 1].plot(it_test_sgd, test_acc_sgd, 'r-', label='SGD', alpha=0.7)
    axes[0, 1].plot(it_test_gda, test_acc_gda, 'b-', label='GDA', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test loss comparison
    axes[1, 0].plot(it_test_sgd, test_losses_sgd, 'r-', label='SGD', alpha=0.7)
    axes[1, 0].plot(it_test_gda, test_losses_gda, 'b-', label='GDA', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test Loss')
    axes[1, 0].set_title('Test Loss Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate comparison
    axes[1, 1].plot(it_train_sgd, lrs_sgd, 'r-', label='SGD (fixed)', alpha=0.7)
    axes[1, 1].plot(it_train_gda, lrs_gda, 'b-', label='GDA (adaptive)', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Comparison')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    plt.show()




if __name__ == "__main__":
    print("=" * 80)
    print("Training ResNet18 with GDA and SGD")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    N_train = 50000
    batch_size = 128
    n_epoch = 10  
    
    print("\n[1] Loading CIFAR-10 dataset...")
    trainloader, testloader, num_classes = load_data(batch_size=batch_size, num_workers=2)
    checkpoint = len(trainloader) // 3 + 1
    print(f"Checkpoint: {checkpoint}")
    
    n_seeds = 1
    max_seed = 424242
    rng = np.random.default_rng(42)
    seeds = [rng.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]
    
    for r, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Running with seed: {seed}")
        print(f"{'='*80}")
        
        
        print("\n[2] Training with GDA...")
        seed_everything(seed)
        net_gda = ResNet18()
        net_gda.to(device)
        
        start_time = time.time()
        results_gda = train_with_gda(
            net=net_gda, 
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            N_train=N_train,
            n_epoch=n_epoch, 
            weight_decay=0,
            sigma=0.1,
            lr=0.2,
            k=0.75,
            checkpoint=checkpoint, 
            batch_size=batch_size, 
            noisy_train_stat=False
        )
        time_gda = time.time() - start_time
        
        print(f"\nGDA Training completed in {time_gda:.2f} seconds")
        print(f"Final test accuracy: {results_gda[3][-1]*100:.2f}%")
        print(f"Best test accuracy: {max(results_gda[3])*100:.2f}%")
        
        
        print("\n[3] Training with SGD...")
        seed_everything(seed)
        net_sgd = ResNet18()
        net_sgd.to(device)
        
        start_time = time.time()
        results_sgd = train_with_sgd(
            net=net_sgd,
            trainloader=trainloader,
            testloader=testloader,
            device=device,
            N_train=N_train,
            n_epoch=n_epoch,
            weight_decay=0,
            checkpoint=checkpoint,
            batch_size=batch_size,
            noisy_train_stat=False
        )
        time_sgd = time.time() - start_time
        
        print(f"\nSGD Training completed in {time_sgd:.2f} seconds")
        print(f"Final test accuracy: {results_sgd[3][-1]*100:.2f}%")
        print(f"Best test accuracy: {max(results_sgd[3])*100:.2f}%")
        
        
        print("\n[4] Saving results...")
        output_dir = 'output'
        figures_dir = os.path.join(output_dir, 'figures')
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        
        method_gda = f'gda_0.2_0.1_0.75'
        experiment = 'cifar10_resnet18'
        save_results(*results_gda, method=method_gda, experiment=experiment, folder=output_dir)
        
        method_sgd = 'sgd_0.2'
        save_results(*results_sgd, method=method_sgd, experiment=experiment, folder=output_dir)
        
       
        print("\n[5] Plotting results...")
        
        gda_plot_path = os.path.join(figures_dir, 'ann_gda_results.png')
        plot_single_result(results_gda, 'GDA', gda_plot_path)
        
        sgd_plot_path = os.path.join(figures_dir, 'ann_sgd_results.png')
        plot_single_result(results_sgd, 'SGD', sgd_plot_path)
        
        comparison_plot_path = os.path.join(figures_dir, 'ann_comparison.png')
        plot_comparison(results_sgd, results_gda, comparison_plot_path)
        
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"GDA:")
        print(f"  - Time: {time_gda:.2f}s")
        print(f"  - Best accuracy: {max(results_gda[3])*100:.2f}%")
        print(f"  - Final learning rate: {results_gda[6][-1]:.6f}")
        print(f"\nSGD:")
        print(f"  - Time: {time_sgd:.2f}s")
        print(f"  - Best accuracy: {max(results_sgd[3])*100:.2f}%")
        print(f"  - Learning rate: {results_sgd[6][0]:.6f} (fixed)")
        print(f"\nAccuracy difference: {(max(results_gda[3]) - max(results_sgd[3]))*100:+.2f}%")
        print("="*80)
        print("\nExperiment completed successfully!")
