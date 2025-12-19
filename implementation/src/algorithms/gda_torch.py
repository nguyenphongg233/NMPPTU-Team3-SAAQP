"""
GDA (Gradient Descent with Adaptive learning rate) optimizer
"""

import torch
import numpy as np


def accuracy_and_loss(net, dataloader, device, criterion):
   
    net.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).cpu().item() / len(dataloader)

    return correct / total, loss


def train_with_gda(net, trainloader, testloader, device, N_train, n_epoch=2, weight_decay=0, 
                   eps=1e-8, sigma=0.1, lr=0.2, k=0.75, checkpoint=125, batch_size=128, 
                   noisy_train_stat=True):
    
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []

    net.train()
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=weight_decay, lr=0.2)

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            e = 0

            for p in net.parameters():
                if p.requires_grad is False:
                    continue
                dp = p.grad
                p_pre = p.data.clone()
                p.data = p.data - lr*dp

                e += torch.matmul(dp.flatten(),(p_pre - p.data).flatten())
                p.grad.zero_()

            dl = criterion(net(inputs), labels)
            if dl.item() - loss.item() + sigma*(e) <= 0:
                lr = lr
            else:
                print('Learning rate gets updated')
                lr = k*lr

            running_loss += loss.item()

            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(lr)

            if i % checkpoint == checkpoint - 1:
                if running_loss / checkpoint < 0.01:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / checkpoint), end='')
                running_loss = 0.0
                test_a, test_l = accuracy_and_loss(net, testloader, device, criterion)
                test_acc.append(test_a)
                test_losses.append(test_l)
                net.train()
                it_test.append(epoch + i * batch_size / N_train)

        if not noisy_train_stat:
            it_train.append(epoch)
            train_a, train_l = accuracy_and_loss(net, trainloader, device, criterion)
            train_acc.append(train_a)
            losses.append(train_l)
            net.train()

    return (np.array(losses), np.array(test_losses), np.array(train_acc), np.array(test_acc),
            np.array(it_train), np.array(it_test), np.array(lrs), np.array(grad_norms))
