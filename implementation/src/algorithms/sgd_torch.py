"""
SGD (Stochastic Gradient Descent) optimizer
"""

import torch
import numpy as np
import copy


def train_with_sgd(net, trainloader, testloader, device, N_train, n_epoch=2, weight_decay=0, 
                   eps=1e-8, checkpoint=125, batch_size=128, noisy_train_stat=True):
   
    losses = []
    train_acc = []
    test_losses = []
    test_acc = []
    it_train = []
    it_test = []
    grad_norms = []

    prev_net = copy.deepcopy(net)
    prev_net.to(device)
    net.train()
    prev_net.train()
    lrs = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), weight_decay=weight_decay, lr=0.2)

    # Import accuracy_and_loss from gda_torch
    from algorithms.gda_torch import accuracy_and_loss

    for epoch in range(n_epoch):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            prev_outputs = prev_net(inputs)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if (i % 10) == 0:
                if noisy_train_stat:
                    losses.append(loss.cpu().item())
                    it_train.append(epoch + i * batch_size / N_train)
                lrs.append(optimizer.param_groups[0]['lr'])

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
                grad_norms.append(np.sum([p.grad.data.norm().item() for p in net.parameters()]))
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
