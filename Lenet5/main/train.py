import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from net import Net
import  matplotlib.pyplot as plt
# 定义超参数
n_epochs = 3  # 训练次数
batch_size_train = 64  # 训练集批量大小
batch_size_test = 1000  # 测试集批量大小
learning_rate = 0.01  # 学习率
momentum = 0.5  # 动量
log_interval = 10  # 每隔多少个batch输出一次信息
random_seed = 1  # 随机种子
torch.manual_seed(random_seed)  # 设置随机种子
# 加载MNIST数据集
"""
torch.utils.data.DataLoader中参数：(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
作用:数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器
    dataset (Dataset) – 加载数据的数据集。
    batch_size (int, optional) – 每个batch加载多少个样本(默认: 1)。
    shuffle (bool, optional) – 设置为True时会在每个epoch重新打乱数据(默认: False).
    sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
    num_workers (int, optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
    collate_fn (callable, optional) –
    pin_memory (bool, optional) –
    drop_last (bool, optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

     数据集：torchvision.datasets.MNIST():dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)
     transform： 传入一个函数，原始图片作为输入，返回一个转换后的图片，就是对图片进行处理，如果需要，那就传入相应的
     函数，你不想对函数进行处理，那就设为false
     torchvision.transforms.Compose()输入进去的是一个list[,]，将多个transform组合起来使用。

     torchvision.transforms.ToTensor()：
     把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
     --->test/test1.py
     torchvision.transforms.Normalize：对数据进行归一化的操作
     
     batch_size:
     shuffle:洗牌

     python 为什么可以在对象后面加括号，即对象()?
     结论：因为在类中写了__call__方法，对象()就是调用__call__()
     详情：
     在python中，一切都是一个对象。创建和调用的函数也是对象。python中任何可以调用的对象都是可调用的对象。
    但是，如果希望python中的类对象是可调用对象，则必须在类内定义__call__方法。
    调用对象时，会调用__call__(self, ...)方法
    任何类都可以定义一个__call__方法来定义它的一个实例被调用的意义
"""
'''
数据集并不是一个图片形式的文件，而是二进制文件，要转化的话，https://blog.csdn.net/qq_44447544/article/details/124131441
'''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=True, download=False,
    # transform=torchvision.transforms.Compose(
    # [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    transform=torchvision.transforms.ToTensor()#这个操作是一定要的，转成tensor 并映射到0,1区间
                                            ),
    batch_size=batch_size_train, shuffle=True)
'''
 这里要说一下，我们通常python读取图片的时候用的pillow，
 这里MINIST数据集中一张图片是H×W×C（长×宽×通道）=28*28*1=758的像素图片，
 其中每个像素点中的信息用[0,255]来表示，但是对于神经网络的训练来说太大了
 ，占用太多算力，所以我们通常将像素点中的信息量压缩成[-1,1]，
 且满足一定均值和方差的正态分布，并且是C×H×W。
 所以在torchvision中的transforms.Compose()函数的作用就是将pillow中的信息转成满足需求的张量。
 
transforms.ToTensor()的作用是变成C×H×W，且将信息量变成【0，1】
transforms.Normalize(0.1307,),(0.3081)的作用是按照一定均值与方差将信息量变为满足一定均值和方差的正态分布。大部分的常用数据集都是有公认的均值与方差的。

'''
# 初始化网络和优化器
#batch_size 就是同时计算多少张图片
#Net():class Net(torch.nn.Module)
#torch.nn.module中的parameters()，返回一个 包含模型所有参数 的迭代器。一般用来当作optimizer的参数。
network = Net()
network.cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate)
#optimizer这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新,在后面会计算出梯度
#然后使用梯度下降算法对权重和偏置进行更新，注意，梯度和权重是不一样的东西，
# 使用打印输出来跟踪进度。为创建良好培训曲线，创建两个列表来节省培训和测试损失。在x轴上显示网络在培训期间看到的培训示例的数量
train_losses = []
train_counter = []
# 每次训练都从上一次训练得到的模型开始
# network_state_dict = torch.load('./model.pth')
# network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load('./optimizer.pth')
# optimizer.load_state_dict(optimizer_state_dict)


def train(epoch):
    network.train()  # 设置网络为训练模式
    #如果是一个可迭代对象，就可以使用enumerate()，同时获得下标和对应的值
    for batch_idx, (data, target) in enumerate(train_loader):#这个for循环的用法还真是不会，batch_idx
        # 其中 data 是一个大小为 [batch_size, C,H,W] 的张量，表示输入数据；
        #target 是一个大小为 [batch_size] 的张量，表示对应的目标数据,也就是标签
        optimizer.zero_grad()  # 清除上一次的梯度
        output = network(data.cuda())  # 获取输出,我这数据是转成了cuda()类型的tensor
        #output=[bactch_size,CxHxW],target=[batch_size]
        loss = F.nll_loss(output, target.cuda())  # 计算损失,负对数似然损失（Negative Log Likelihood Loss）
        loss.backward()  # 反向传播 就是计算梯度的，和其他的没有关系
        #计算当前张量 loss 中元素的梯度，并将梯度存储在各个参与计算的张量的 .grad 属性中。具体来说，
        #这个方法会对所有与 loss 张量有关的可训练参数进行求导，并将结果累加到它们的 .grad 属性中
        optimizer.step()  # 更新参数，如何更新参数，就有不同的算法，SDG等
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            #batch_idx * len(data)表示已经训练了多少张图片
            #len(train_loader.dataset) 总图片数
            #100. * batch_idx / len(train_loader)这个东西不知道是什么
            print(len(data))#就输出第0维的大小，也就是batch_size
            print(len(train_loader.dataset))
            train_losses.append(loss.item())

            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


if __name__ == '__main__':
    #这里原本的数据就是小数，灰度级不是[0,255],这里的灰度级是[0,1]
    # for batch_idx, (data, target) in enumerate(train_loader):
    # images, labels = next(iter(train_loader))
    # print(images.size())  #torch.Size([64, 1, 28, 28])
    # print(images[0])
    # plt.imshow((images[0]).permute(1, 2, 0), cmap='gray')
    # plt.show()
    '''
    数据加载器中数据的维度是[B, C, H, W]，我们每次只拿一个数据出来就是[C, H, W]，
    而matplotlib.pyplot.imshow要求的输入维度是[H, W, C]，
    所以我们需要交换一下数据维度，把通道数放到最后面，这里用到pytorch里面的permute方法
    '''
    for epoch in range(1, n_epochs + 1):
        train(epoch)
