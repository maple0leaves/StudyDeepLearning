import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from net import Net

# 定义超参数
n_epochs = 3  # 训练次数
batch_size_train = 64  # 训练集批量大小,
batch_size_test = 1000  # 测试集批量大小
learning_rate = 0.01  # 学习率
momentum = 0.5  # 动量
log_interval = 10  # 每隔多少个batch输出一次信息
random_seed = 1  # 随机种子
torch.manual_seed(random_seed)  # 设置随机种子
# 加载MNIST数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ])),
    batch_size=batch_size_test, shuffle=True)
# 初始化网络和优化器
network = Net()
network.cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate)
# 使用打印输出来跟踪进度。为创建良好培训曲线，创建两个列表来节省培训和测试损失。在x轴上显示网络在培训期间看到的培训示例的数量
test_losses = []

# 进入测试循环，运行测试损失和精度。
# 使用训练得到的模型和优化器来测试
network_state_dict = torch.load('./model.pth')
network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('./optimizer.pth')
optimizer.load_state_dict(optimizer_state_dict)


def test():
    network.eval()  # 设置网络为测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.cuda())
            test_loss += F.nll_loss(output, target.cuda(), size_average=False).item()
            #output.data是用来获取数据的
            #torch.Tensor.data属性返回一个指向当前张量数据的新张量，但是不会计算梯度。通常情况下，
            # 我们使用该属性获取一个张量的数值数据，而不需要为其计算梯度，这可以提高效率并减少内存占用
            '''
            torch.max是一个PyTorch函数，用于返回张量中最大值和指定轴上的最大值索引。
            函数定义如下：
            torch.max(input, dim=None, keepdim=False, out=None) -> Tuple[Tensor, Tensor]
            其中，参数含义如下：
            input：输入的张量。
            dim：指定操作的维度，如果不指定，则默认返回整个张量中的最大值。可以是单个整数或元组。
            keepdim：是否保留输出张量的维度信息，默认为 False。
            out：输出张量，如果给出，则输出结果会被写入此张量并返回，否则会创建新的张量返回。
            返回值为一个元组 (values, indices)，其中 values 是最大值的张量， indices 是对应的最大值索引的张量。
            '''
            x= torch.max(output, dim=1)
            pred = torch.max(output, dim=1)[0].cpu().numpy()
            print(pred)
            # 直接这样也行，不用.data
            #
            # pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.cuda().data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    # 进入训练循环，运行训练损失和精度。
    for epoch in range(1, n_epochs + 1):
        test()
