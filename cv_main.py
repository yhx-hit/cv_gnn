import matplotlib.pyplot as plt
import numpy as np
import gzip, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu
from torchvision import datasets, transforms


def complex_relu(input_r, input_i):
    return relu(input_r), relu(input_i)


class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i) + self.conv_i(input_r)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r)


class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features, 3))
            self.bias = Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:, :2], 1.4142135623730951)
            init.zeros_(self.weight[:, 2])
            init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        assert (len(input_r.shape) == 4)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])

            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
                           Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]

        return input_r, input_i


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 20, 5, 2)
        self.bn = ComplexBatchNorm2d(20)
        self.conv2 = ComplexConv2d(20, 50, 5, 2)
        self.fc1 = ComplexLinear(4 * 4 * 50, 500)
        self.fc2 = ComplexLinear(500, 10)

        self.bn4imag = BatchNorm2d(1)
        self.conv4imag = Conv2d(1, 1, 3, 1, padding=1)

    def forward(self, x):
        xr = x
        # imaginary part BN-ReLU-Conv-BN-ReLU-Conv as shown in paper
        xi = self.bn4imag(xr)
        xi = relu(xi)
        xi = self.conv4imag(xi)

        # flow into complex net
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)

        xr, xi = self.bn(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        #         print(xr.shape)
        xr = xr.reshape(-1, 4 * 4 * 50)
        xi = xi.reshape(-1, 4 * 4 * 50)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return F.log_softmax(x, dim=1)


batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# class DealDataset(Dataset):
#     """
#         读取数据、初始化数据
#     """
#
#     def __init__(self, folder, data_name, label_name, transform=None):
#         (train_set, train_labels) = load_data(folder, data_name,
#                                               label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
#         self.train_set = train_set
#         self.train_labels = train_labels
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img, target = self.train_set[index], int(self.train_labels[index])
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target
#
#     def __len__(self):
#         return len(self.train_set)
#
#
# def load_data(data_folder, data_name, label_name):
#     with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
#         y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
#
#     with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
#         x_train = np.frombuffer(
#             imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
#     return (x_train, y_train)
#
#
# train_dataset = DealDataset(r'/home/yyyyyyhx/Downloads/CV-GMTINet-pytorch/data', "train-images-idx3-ubyte.gz",
#                            "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())
# test_dataset = DealDataset(r'/home/yyyyyyhx/Downloads/CV-GMTINet-pytorch/data', "t10k-images-idx3-ubyte.gz",
#                            "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())


train_dataset = datasets.MNIST('/home/yyyyyyhx/Downloads/CV-GMTINet-pytorch/mnistdata', train=True, transform=trans, download=False)
test_dataset = datasets.MNIST('/home/yyyyyyhx/Downloads/CV-GMTINet-pytorch/mnistdata', train=False, transform=trans, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ComplexNet().to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# train steps
train_loss = []
for epoch in range(50):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )

plt.plot(train_loss)