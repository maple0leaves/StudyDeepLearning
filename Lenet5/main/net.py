import torch
import matplotlib.pyplot as plt
# -- coding: utf-8 --
'''
torch.nn.Module
所有网络的基类。
你的模型也应该继承这个类。
Modules也可以包含其它Modules,允许使用树结构嵌入他们。你可以将子模块赋值给模型属性。
evaluation:评价
'''
class Net(torch.nn.Module):
    """
    Module中方法介绍：
    torch.nn.Conv2d(1, 10, kernel_size=5)
    通过上面方式赋值的submodule会被注册。当调用 .cuda() 的时候，submodule的参数也会转换为cuda Tensor
    forward(* input)
    定义了每次执行的 计算步骤。 在所有的子类中都需要重写这个函数。
    eval()
    将模型设置成evaluation模式
    仅仅当模型中有Dropout和BatchNorm是才会有影响
    将module设置为 training mode。
    仅仅当模型中有Dropout和BatchNorm是才会有影响。
    load_state_dict(state_dict)
    将state_dict中的parameters和buffers复制到此module和它的后代中。state_dict中的key必须和 model.state_dict()返回的key一致。
     NOTE：用来加载模型参数。
     modules()
    返回一个包含 当前模型 所有模块的迭代器，可以循环一下看模型的结构
    parameters(memo=None)
    返回一个 包含模型所有参数 的迭代器。
    一般用来当作optimizer的参数
    torch.nn.Sequential(* args)：
    一个时序容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict。
    torch.nn.Conv2d:
    Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    前3个必填，其他都有默认值
    简单点的输出图片边长：  N = (W − F + 2P ) / S + 1
    N：输出边长
    W：输入边长，F：卷积核边长，P：填充大小，S:步长
    当通过N = (W − F + 2P ) / S + 1计算式得到的输出尺寸非整数时，会通过删除多余的行和列来保证卷积的输出尺寸为整数。

    in_channels ([int]) - 输入图像的通道数
    out_channels ([int]) - 输出通道数，也是卷积核个数
    kernel_size ([int] or *[tuple]) - 卷积核尺寸
    stride ([int] or [tuple], optional ) - 步长，默认为1
    padding ([int] [tuple] or [str] optional ) - 扩边，默认为0
    padding_mode ([str] optional ) - zeros, reflect, replicate, 默认为 zeros.
    dilation ([int] [tuple] optional ) - 膨胀核，默认为1
    groups ([int] optional ) - 分组，默认为1
    bias ([bool] optional ) - 如果设置为 true, 添加学习的 bias
    https://blog.csdn.net/zhoujinwang/article/details/129270062
   """
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 10, kernel_size=5),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2)
        # )

        self.c1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.r1= torch.nn.ReLU()
        self.m1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            # 激活函数解释：https://www.zhihu.com/question/22334626
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),#对输入数据做线性变换：y=Ax+b,不是激活函数那种非线性变换了
            torch.nn.Linear(50, 10),
            #感觉就是起到一种把输出降下来的作用
        )

    def forward(self, x):
        batch_size = x.size(0)
        #@author 221 2023年4月12日
        #我还有个问题，为什么输入图象里面的值是负数？？灰度级不是不能为负吗？？？后面的卷积也有负数，这是为什么？
        #可能是因为之前标准化了，torchvision.transforms.Normalize((0.1307,), (0.3081,))
        #1.8503e-01 这个是科学计数法，即1.8503x10^-1，
        #在指数为负数时，可以在指数前面添加一个 "0" 来使表达更清晰，因此 1.8503e-1 可以写成 1.8503e-01
        #这个e就是10^
        print('一开始读取的图片：',x)
        #(batch_size ,channle,H,W)
        #每一个卷积核，这个卷积核的深度是和图片对应的，
        # 都要过一遍这个图象，生成一个通道
        print("一开始读取图片的形状",x.shape)#torch.Size([64, 1, 28, 28])
        x= self.c1(x)

        print('卷积后',x)#卷积后在[-1,1]
        print("卷积后的形状",x.shape)#torch.Size([64, 10, 24, 24])
        x= self.r1(x)#由于卷积只能学习到线性特新征，激活函数可以将任何输入值映射到一个新的指定范围内，让模型学习到非线性特征，
        # 增强表达能力，并且可以防止过度拟合或欠拟合的情况发生
        print("激活后",x)
        x= self.m1(x)#池化减少参数量，缩小尺寸，池化不像卷积一行一行移动的，池化是卷积核在移动过程中不会重叠，卷积是卷积核移动过程中会重叠
        #池化kernel_size=stride，卷积各论个的
        print('池化后',x)

        # x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次


        print(x.shape)
        #这个展平操作可以用torch.nn.flatten()替代
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        #这-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成xxx列
        #不知道是按行展开，还是按列展开？？？
        # 是一行一行取的,具体看test/test3.py
        print(x.shape)
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
        #全连接解释
    '''
    过拟合：
    当模型过拟合时，它可能会记住训练数据中的噪声或者不相关的特征，而忽略真正有用的特征。
    这将导致模型泛化能力差，即对新的数据表现不佳
    '''