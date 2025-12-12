

import torch
import torch.nn as nn
import numpy as np

class SGDTorch:
    def __init__(self, net, lr=0.2, weight_decay=0, momentum=0):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.optimizer = torch.optim.SGD(
            net.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            momentum=momentum
        )
        self.lr_history = []
        
    def step(self, inputs, labels, criterion):
    
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        self.lr_history.append(self.lr)
        
        return loss.item()
    
    def get_lr(self):
        """Get current learning rate"""
        return self.lr
    
    def get_lr_history(self):
        """Get learning rate history"""
        return np.array(self.lr_history)


def train_with_sgd(net, trainloader, testloader, device, n_epoch=2, 
                   lr=0.2, weight_decay=0, momentum=0,
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
    sgd_optimizer = SGDTorch(net, lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    N_train = len(trainloader.dataset)
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # SGD step
            loss = sgd_optimizer.step(inputs, labels, criterion)
            
            running_loss += loss
            
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss)
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(sgd_optimizer.get_lr())
            
            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint))
                
                running_loss = 0.0
                
                from algorithms.gda_torch import accuracy_and_loss
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                
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
            from algorithms.gda_torch import accuracy_and_loss
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
