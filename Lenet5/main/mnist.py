import torch
import matplotlib.pyplot as plt
from test import test_loader
from net import Net
import torch.optim as optim

'''
先训练起来
'''
# 定义超参数
learning_rate = 0.01  # 学习率
momentum = 0.5  # 动量
# # 查看测试数据组成
examples = enumerate(test_loader)#生成枚举对象
batch_idx, (example_data, example_targets) = next(examples)#把枚举对象放进去迭代
# # 绘制一些测试数据
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# # 测试训练前的准确率
# with torch.no_grad():
#     output = network(example_data.cuda())
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# # 绘制训练曲线
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Training Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
network = Net()
network.cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate)
#随机的意思是batch_size 中的图片是随机的，梯度下降
# ↑↑ 随机梯度下降 SGD
# network_state_dict = torch.load('./model.pth')
# network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load('./optimizer.pth')#优化？？？
# optimizer.load_state_dict(optimizer_state_dict)
# 测试模型准确率
with torch.no_grad():
    output = network(example_data.cuda())#在这没有使用softmax转换成概率，直接取最大值
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
