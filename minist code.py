# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:30:04 2020

@author: Lenovo
"""

from __future__ import print_function
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_mnist(path='D:/fashion-mnist', kind1='train',kind2='t10k'):
 labels_path = os.path.join(path,
 '%s-labels-idx1-ubyte.gz'
 % kind1)
 images_path = os.path.join(path,
 '%s-images-idx3-ubyte.gz'
 % kind1)
 with gzip.open(labels_path, 'rb') as lbpath:
  labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
 offset=8)
 with gzip.open(images_path, 'rb') as imgpath:
  images = np.frombuffer(imgpath.read(), dtype=np.uint8,
 offset=16).reshape(len(labels),1,28,28)
 labels = torch.from_numpy(labels).type(torch.LongTensor)
 images = torch.from_numpy(images).type(torch.float32)
 trainset = torch.utils.data.TensorDataset(images, labels)

 labels_path = os.path.join(path,
 '%s-labels-idx1-ubyte.gz'
 % kind2)
 images_path = os.path.join(path,
 '%s-images-idx3-ubyte.gz'
 % kind2)
 with gzip.open(labels_path, 'rb') as lbpath:
  labels1 = np.frombuffer(lbpath.read(), dtype=np.uint8,
  offset=8)
 with gzip.open(images_path, 'rb') as imgpath:
  images = np.frombuffer(imgpath.read(), dtype=np.uint8,
  offset=16).reshape(10000,1 ,28,28)
 t_labels = torch.from_numpy(labels1).type(torch.LongTensor)
 t_images = torch.from_numpy(images).type(torch.float32)
 testset = torch.utils.data.TensorDataset(t_images, t_labels)
 return {1:trainset,2:testset,3:labels1}


class fmnistNet(nn.Module):
    def __init__(self):
        super(fmnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)  #nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0))
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(10816, 256)
        self.fc2=nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        output = F.log_softmax(x, dim=1)#损失函数
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pre_lab=np.array([0])
    target1=np.array([0])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = torch.argmax(output,1)  #得到最可能的预测值
            correct += pred.eq(target.view_as(pred)).sum().item()
            pre_lab = np.append(pre_lab,pred.cpu().numpy().T,axis=0)
            target1=np.append(target1,target.cpu().numpy().T,axis=0)
            #pre_lab=pred.cpu().numpy().(1,16)


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return {1:pre_lab,2:target1}

    #通过图形表示
def viewer(pre_lab,target):
    plt.figure(figsize=(12,12))
    class_label=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    test_data_y = pre_lab[1:10001,]
    target=target[1:10001,]
    conf_mat = confusion_matrix(test_data_y,target)
    df_cm = pd.DataFrame(conf_mat, index = class_label,columns = class_label)
    ht= sns.heatmap(df_cm,annot = True, fmt='d', linewidths=.5,cmap = "YlGnBu")
    ht.yaxis.set_ticklabels(ht.yaxis.get_ticklabels(),rotation = 0, ha = "right")
    plt.ylabel('True label')
    plt.xlabel("Predicted label")
    plt.show()

def main():
    # 训练集参数设置
    parser = argparse.ArgumentParser(description='PyTorch MNIST/Fashion Mnist Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    kwargs = {'batch_size': args.batch_size}
    kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},)

    dataset=load_mnist()
    train_loader = torch.utils.data.DataLoader(dataset.get(1),**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset.get(2), **kwargs)
    #训练
    model = fmnistNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        x=test(model, device, test_loader)
    viewer(x.get(1),x.get(2))
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    return x


if __name__ == '__main__':
    x=main()
