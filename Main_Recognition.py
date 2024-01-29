# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:36:33 2020

@author: Lenovo
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse 
import torch.optim as optim

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1=nn.Conv2d(1,32,5,2,1)
        self.pool=nn.MaxPool2d(3,2)
        self.conv2=nn.Conv2d(32,64,3,1,1)
        self.conv3=nn.Conv2d(64,128,3,1,1)
        self.conv4=nn.Conv2d(128,64,3,1,1)
        self.fc1=nn.Linear(3136,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,10)
    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.pool(x)
        x = torch.flatten(x, 1)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.fc3(x)
        return x

#残差块（1*1压缩，3*3卷积，1*1维数恢复）
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)#优化卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out        
    
#网络主体（卷积（11*11，2）-卷积（3*3，3）-maxpool（3，2）-残差层（4）-全连接）
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=500):
        self.inplanes =32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=11, stride=3,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        #self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=3,padding=1,bias=False)
        #self.bn2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2,2)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2,2)
        self.fc = nn.Linear(256 * block.expansion,500)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


parse = argparse.ArgumentParser(description='Params for training. ')
# 数据集根目录，根据自己数据的位置进行修改，应该是root的下级文件夹就有train和test文件夹
parse.add_argument('--root', type=str, default='/home/aiserver1/Desktop/deuxime/data/OCR/char', help='path to data set')
# 模式，3选1，当default为train时，就会运行下面的trian函数，就只进行模型训练
parse.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'inference', 'inference_all'])
# checkpoint 路径，用来保存模型训练的状态
parse.add_argument('--log_path', type=str, default=os.path.abspath('.') + '/log.pth', help='dir of checkpoints')
# 第一次训练时，需要将restore设为False，等到代码运行一段时间生成log.pth文件后，停止训练，将其改为True，再继续运行即可
parse.add_argument('--restore', type=bool, default=True, help='whether to restore checkpoints')

parse.add_argument('--batch_size', type=int, default=16, help='size of mini-batch')
parse.add_argument('--image_size', type=int, default=64, help='resize image')
# 迭代次数
parse.add_argument('--epoch', type=int, default=5)
# 数据集类别数是3755，所以给定了一个选择范围，只对100个类别的数据进行训练和预测
parse.add_argument('--num_class', type=int, default=500)
args = parse.parse_args()  # 解析参数

#根据数据集编写训练和测试的标签索引
def classes_txt(root, out_path, num_class=None):
    """
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    """
    dirs = os.listdir(root)  # 列出根目录下所有类别所在文件夹名
    if not num_class:		# 不指定类别数量就读取所有
        num_class = len(dirs)
    if not os.path.exists(out_path):  # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
    # 如果文件中本来就有一部分内容，只需要补充剩余部分
    # 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join("%s/%s/%s" % (root, dir, file)) + '\n')


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:  # 只读取前 num_class 个类
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        """
        :param index: 索引
        :return: 获取图片和标签
        """
        image = Image.open(self.images[index]).convert('RGB')  # 用PIL.Image读取图像
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        """
        :return:获取数据集长度
        """
        return len(self.labels)
    
def t_validation():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    #model=network()
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

            if i % 10000 == 9999:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))
    
    
def validation():
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    test_set = MyDataset(args.root + '/test.txt', num_class=args.num_class, transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    #model=network()
    model.to(device)

    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += sum(int(predict == labels)).item()
            # 根据评论区反馈，如果上面这句报错，可以换成下面这句试试：
            correct += (predict == labels).sum().item()

            if i % 10000 == 9999:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))
    
    
def train():
	# 由于我的数据集图片尺寸不一，因此要进行resize，这里还可以加入数据增强，灰度变换，随机剪切等等
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    train_set = MyDataset(args.root + '/train.txt', num_class=args.num_class, transforms=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	# 选择使用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #model=network()
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.to(device)
	# 训练模式
    model.train()

    criterion = nn.CrossEntropyLoss()#交叉熵计算损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)#Adam优化
	# 由命令行参数决定是否从之前的checkpoint开始训练
    if args.restore:
        checkpoint = torch.load(args.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        loss = 0.0
        epoch = 0
    while epoch < args.epoch:
        running_loss = 0.0
        for i, data in enumerate(train_loader):
        # 这里取出的数据就是 __getitem__() 返回的数据
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()#梯度置零
            outs = model(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 1000 == 999:  # every 200 steps
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss /1000))
                running_loss = 0.0
		# 保存 checkpoint
        if epoch % 1 == 0:
            print('Save checkpoint...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                       args.log_path)
            t_validation()
            validation()
        epoch += 1
    print('Finish training')



def inference():
    print('Start inference...')
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])

    f = open(args.root + '/test.txt')
    num_line = sum(line.count('\n') for line in f)
    f.seek(0, 0)
    # 在文件中随机取一个路径
    line = int(torch.rand(1).data * num_line - 10) # -10 for '\n's are more than lines
    while line > 0:
        f.readline()
        line -= 1
    img_path = f.readline().rstrip('\n')
    f.close()
    label = int(img_path.split('/')[-2])
    print('label:\t%4d' % label)
    input = Image.open(img_path).convert('RGB')
    input = transform(input)
    # 网络默认接受4维数据，即[Batch, Channel, Heigth, Width]，所以要加1个维度
    input = input.unsqueeze(0)
    #model = ResNet(Bottleneck, [3, 4, 6, 3])
    model=network()
    model.eval()
    checkpoint = torch.load(args.log_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    output = model(input)
    _, pred = torch.max(output.data, 1)
    
    print('predict:\t%4d' % pred)

if __name__ == '__main__':
    classes_txt(args.root + '/train', args.root + '/train.txt', num_class=args.num_class)
    classes_txt(args.root + '/test', args.root + '/test.txt', num_class=args.num_class)

    if args.mode == 'train':
        train()
    elif args.mode == 'validation':
        validation()
    elif args.mode == 'inference':
        inference()

