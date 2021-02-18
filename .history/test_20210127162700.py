import torch
import matplotlib.pyplot as plt


# 建立一个三层的神经网络
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__()

        self.hidden = torch.nn.Linear(n_features, n_hidden)  # 定义隐藏层：线性
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 定义输出层：线性

    def forward(self, x):  # 正向传播
        a = torch.relu(self.hidden(x))
        b = self.predict(a)
        return b


if __name__ == '__main__':
	
	# 随机生成一些点（x,y）
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())

    # 画图
    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    net = Net(1, 10, 1)
    print(net)

    plt.ion()
    plt.show()

    # 训练工具：传入net的所有参数，设置学习率
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    # 损失函数：均方差
    loss_function = torch.nn.MSELoss()

    for t in range(1000):
        prediction = net(x)  # 输入x，输出预测值

        loss = loss_function(prediction, y)  # 计算预测值和真实值之间的误差

        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()        # 误差反向传播, 计算参数更新值
        optimizer.step()       # 将参数更新值施加到 net 的 parameters 上

        if t % 5 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(),
                     fontdict={'size': 15, 'color': 'red'})
            plt.pause(0.1)