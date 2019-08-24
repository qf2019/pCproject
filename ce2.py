import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.cuda

trainset = torchvision.datasets.CIFAR10(root='/home/dell/qf/pj1/cifar',train=True,download=False, transform=
                        transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/dell/qf/pj1/cifar',train=True,download= False, transform=
                        transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ResBlk(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.blk1 = ResBlk(64, 128)
        self.blk2 = ResBlk(128, 256)
        self.blk3 = ResBlk(256, 512)
        self.blk4 = ResBlk(512, 512)
        # self.pooler = nn.MaxPool2d(kernel_size=4, stride=2,padding=0)
        self.outlayer = nn.Linear(512*32*32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # [b, 64, 32, 32]
        x = self.blk1(x)
        # [b, 128, 32, 32]
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # [b, 512, 32, 32]
        # x = self.pooler(x)
        # print(x.shape)
        x = x.view(-1, 512*32*32)
        x = self.outlayer(x)
        return x


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = 'model.pth'
    net = Resnet18().to(device)
    loss_fun = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net = net.train()
    EPOCH = 1
    running_loss = 0
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': EPOCH,
    }
    for epoch in range(EPOCH):
        for step, data in enumerate(trainloader):
            # 1.数据的预处理操作： 将数据进行Variable
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            # 2.将优化器进行梯度的清零
            optimizer.zero_grad()
            # 3. 使用网络得出结果
            output = net(inputs)
            # 4. 得到loss 的值，进行梯度的反向传播
            loss = loss_fun(output, labels)
            loss.backward()
            # 5.利用优化器对所有的参数进行更新
            optimizer.step()
            running_loss += loss
            if step % 2000 == 1999:
                print('[%d, %d ], loss:%5f' % (epoch+1, step+1, running_loss/2000))
                running_loss = 0
        torch.save(state, model_file)
        checkpoint = torch.load(model_file)
        print(checkpoint['optimizer'])
    net = net.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for step, data in enumerate(testloader):
            images, labels = data
            labels = labels.to(device)
            outputs = net(Variable(images.to(device)))
            # loss_test += criterion(outputs, labels).item()
            # _, predicted = t.max(outputs.data, 1)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            step += 1
        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
        # for step, data in enumerate(testloader):
        #     test_input, test_label = data
        #     test_input = Variable(test_input)
        #     test_output = net(test_input)
        #     predicted = test_output.argmax(dim=1)
        #     total += test_label.size(0)
        #     # print(total)
        #     correct += (predicted == test_label).sum()
        #     # print(corret)
        # print('acc:%d %%' % (100 * (correct / total)))