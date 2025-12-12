import torch
import torch.nn as nn
import numpy as np


class GDATorch:
    
    def __init__(self, net, lr=0.2, sigma=0.1, k=0.75):
       
        self.net = net
        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.lr_history = []
        
    def step(self, inputs, labels, criterion):
      
        # Forward pass
        outputs = self.net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # GDA update with learning rate adaptation
        e = 0
        for p in self.net.parameters():
            if p.requires_grad is False:
                continue
            
            dp = p.grad
            p_pre = p.data.clone()
            p.data = p.data - self.lr * dp
            
            e += torch.matmul(dp.flatten(), (p_pre - p.data).flatten())
            p.grad.zero_()
        
        # Check GDA condition and update learning rate
        dl = criterion(self.net(inputs), labels)
        if dl.item() - loss.item() + self.sigma * e <= 0:
            # Condition satisfied, keep learning rate
            pass
        else:
            # Condition not satisfied, reduce learning rate
            self.lr = self.k * self.lr
            # print(f'Learning rate updated to: {self.lr}')
        
        self.lr_history.append(self.lr)
        
        return loss.item()
    
    def get_lr(self):
        """Get current learning rate"""
        return self.lr
    
    def get_lr_history(self):
        """Get learning rate history"""
        return np.array(self.lr_history)


def train_with_gda(net, trainloader, testloader, device, n_epoch=2, 
                   lr=0.2, sigma=0.1, k=0.75, weight_decay=0,
                   checkpoint=125, batch_size=128, noisy_train_stat=True):
   
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []
    lrs = []
    
    net.to(device)
    net.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    gda_optimizer = GDATorch(net, lr=lr, sigma=sigma, k=k)
    
    N_train = len(trainloader.dataset)
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # GDA step
            loss = gda_optimizer.step(inputs, labels, criterion)
            
            running_loss += loss
            
            # Log training statistics
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss)
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(gda_optimizer.get_lr())
            
            # Checkpoint evaluation
            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                
                running_loss = 0.0
                
                # Evaluate on test set
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                
                # Compute gradient norm
                grad_norm = 0.0
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm().item()
                grad_norms.append(grad_norm)
                
                net.train()
                it_test.append(epoch + i * batch_size / N_train)
        
        # End of epoch statistics
        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()
    
    return {
        'losses': np.array(losses),
        'test_losses': np.array(test_losses),
        'train_acc': np.array(train_acc),
        'test_acc': np.array(test_acc),
        'it_train': np.array(it_train),
        'it_test': np.array(it_test),
        'lrs': np.array(lrs),
        'grad_norms': np.array(grad_norms)
    }


def accuracy_and_loss(net, dataloader, device, criterion):
    
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return accuracy, avg_loss
