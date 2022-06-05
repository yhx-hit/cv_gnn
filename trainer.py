import os
# from cv_main import ComplexConv2d, ComplexBatchNorm2d, ComplexLinear, complex_relu
from torch.autograd import Variable
from time import time
from gnn import GNN_module
# import matplotlib.pyplot as plt
import numpy as np
import gzip, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu ,dropout
from torchvision import datasets, transforms
from torchsummary import summary
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d
from complexFunctions import complex_relu, complex_max_pool2d, complex_dropout


class Complex_dropout(Module):
    def __init__(self,p=0.5,training=True):
        super(Complex_dropout, self).__init__()
        self.p = p
        self.training = training

    def forward(self, inputr,inputi):
        # need to have the same dropout mask for real and imaginary part,
        # this not a clean solution!
        #mask = torch.ones_like(input).type(torch.float32)
        input = torch.complex(inputr,inputi)
        mask = torch.ones(*input.shape, dtype = torch.float32)
        mask = dropout(mask, self.p, self.training)*1/(1-self.p)
        mask.type(input.dtype)
        return (mask.cuda()*input).real, (mask.cuda()*input).imag


class complex_relu(Module):

    def __init__(self):
        super(complex_relu, self).__init__()
        self.complexrelu = relu

    def forward(self, input_r, input_i):
        return self.complexrelu(input_r), self.complexrelu(input_i)

class complex_MaxPool2d(Module):

    def __init__(self):
        super(complex_MaxPool2d, self).__init__()
        self.complexMaxPool2d = nn.MaxPool2d(2)

    def forward(self, input_r, input_i):
        return self.complexMaxPool2d(input_r), self.complexMaxPool2d(input_i)

# def complex_relu(input_r, input_i):
#     return relu(input_r), relu(input_i)
#
# def complex_MaxPool2d(n, input_r, input_i):
#     return nn.MaxPool2d(n)(input_r), nn.MaxPool2d(n)(input_i)

class ComplexConv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_r, input_i):
        assert (input_r.size() == input_i.size())
        # return self.conv_r(input_r), self.conv_r(input_i)
        return self.conv_r(input_r) - self.conv_i(input_i), self.conv_r(input_i) + self.conv_i(input_r)
#
#
class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input_r, input_i):
        return self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r)
#
#
# class _ComplexBatchNorm(Module):
#
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(_ComplexBatchNorm, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(num_features, 3))
#             self.bias = Parameter(torch.Tensor(num_features, 2))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(num_features, 2))
#             self.register_buffer('running_covar', torch.zeros(num_features, 3))
#             self.running_covar[:, 0] = 1.4142135623730951
#             self.running_covar[:, 1] = 1.4142135623730951
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_covar', None)
#             self.register_parameter('num_batches_tracked', None)
#         self.reset_parameters()
#
#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_covar.zero_()
#             self.running_covar[:, 0] = 1.4142135623730951
#             self.running_covar[:, 1] = 1.4142135623730951
#             self.num_batches_tracked.zero_()
#
#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             init.constant_(self.weight[:, :2], 1.4142135623730951)
#             init.zeros_(self.weight[:, 2])
#             init.zeros_(self.bias)
#
#
# class ComplexBatchNorm2d(_ComplexBatchNorm):
#
#     def forward(self, input_r, input_i):
#         assert (input_r.size() == input_i.size())
#         assert (len(input_r.shape) == 4)
#         exponential_average_factor = 0.0
#
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         if self.training:
#
#             # calculate mean of real and imaginary part
#             mean_r = input_r.mean([0, 2, 3])
#             mean_i = input_i.mean([0, 2, 3])
#
#             mean = torch.stack((mean_r, mean_i), dim=1)
#
#             # update running mean
#             with torch.no_grad():
#                 self.running_mean = exponential_average_factor * mean \
#                                     + (1 - exponential_average_factor) * self.running_mean
#
#             input_r = input_r - mean_r[None, :, None, None]
#             input_i = input_i - mean_i[None, :, None, None]
#
#             # Elements of the covariance matrix (biased for train)
#             n = input_r.numel() / input_r.size(1)
#             Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
#             Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
#             Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])
#
#             with torch.no_grad():
#                 self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
#                                            + (1 - exponential_average_factor) * self.running_covar[:, 0]
#
#                 self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
#                                            + (1 - exponential_average_factor) * self.running_covar[:, 1]
#
#                 self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
#                                            + (1 - exponential_average_factor) * self.running_covar[:, 2]
#
#         else:
#             mean = self.running_mean
#             Crr = self.running_covar[:, 0] + self.eps
#             Cii = self.running_covar[:, 1] + self.eps
#             Cri = self.running_covar[:, 2]  # +self.eps
#
#             input_r = input_r - mean[None, :, 0, None, None]
#             input_i = input_i - mean[None, :, 1, None, None]
#
#         # calculate the inverse square root the covariance matrix
#         det = Crr * Cii - Cri.pow(2)
#         s = torch.sqrt(det)
#         t = torch.sqrt(Cii + Crr + 2 * s)
#         inverse_st = 1.0 / (s * t)
#         Rrr = (Cii + s) * inverse_st
#         Rii = (Crr + s) * inverse_st
#         Rri = -Cri * inverse_st
#
#         input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
#                            Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r
#
#         if self.affine:
#             input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
#                                                                                None] * input_i + \
#                                self.bias[None, :, 0, None, None], \
#                                self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
#                                                                                None] * input_i + \
#                                self.bias[None, :, 1, None, None]
#
#         return input_r, input_i
#
#
# class ComplexNet(nn.Module):
#
#     def __init__(self):
#         super(ComplexNet, self).__init__()
#         self.conv1 = ComplexConv2d(1, 20, 5, 2)
#         self.bn = ComplexBatchNorm2d(20)
#         self.conv2 = ComplexConv2d(20, 50, 5, 2)
#         self.fc1 = ComplexLinear(4 * 4 * 50, 500)
#         self.fc2 = ComplexLinear(500, 10)
#
#         self.bn4imag = BatchNorm2d(1)
#         self.conv4imag = Conv2d(1, 1, 3, 1, padding=1)
#
#     def forward(self, x):
#         xr = x
#         # imaginary part BN-ReLU-Conv-BN-ReLU-Conv as shown in paper
#         xi = self.bn4imag(xr)
#         xi = relu(xi)
#         xi = self.conv4imag(xi)
#
#         # flow into complex net
#         xr, xi = self.conv1(xr, xi)
#         xr, xi = complex_relu(xr, xi)
#
#         xr, xi = self.bn(xr, xi)
#         xr, xi = self.conv2(xr, xi)
#         xr, xi = complex_relu(xr, xi)
#         #         print(xr.shape)
#         xr = xr.reshape(-1, 4 * 4 * 50)
#         xi = xi.reshape(-1, 4 * 4 * 50)
#         xr, xi = self.fc1(xr, xi)
#         xr, xi = complex_relu(xr, xi)
#         xr, xi = self.fc2(xr, xi)
#         # take the absolute value as output
#         x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
#         return F.log_softmax(x, dim=1)


def np2cuda(array):
    tensor = torch.from_numpy(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name, map_location=lambda storage, loc: storage))
    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

###############################################################
## Vanilla CNN model, used to extract visual features

class EmbeddingCNN(myModel):

    def __init__(self, image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers):
        super(EmbeddingCNN, self).__init__()

        module_list = []
        dim = cnn_hidden_dim
        for i in range(cnn_num_layers):
            if i == 0:
                module_list.append(ComplexConv2d(1, dim, 3, 1,1))
                # module_list.append(Complex_dropout(p=0.1))
                module_list.append(ComplexBatchNorm2d(dim))

            else:
                module_list.append(ComplexConv2d(dim, dim*2, 3, 1,1))
                # module_list.append(Complex_dropout(p=0.1))
                module_list.append(ComplexBatchNorm2d(dim*2))

                dim *= 2
            module_list.append(complex_MaxPool2d())
            module_list.append(complex_relu())
            image_size //= 2
        module_list.append(ComplexConv2d(dim, cnn_feature_size, image_size, 1))
        # module_list.append(Complex_dropout(p=0.1))
        module_list.append(ComplexBatchNorm2d(cnn_feature_size))
        module_list.append(complex_relu())


        self.module_list = nn.ModuleList(module_list)

    def forward(self, inputs):
        xr, xi = inputs.real, inputs.imag
        # x = inputs
        for l in self.module_list:
            xr,xi = l(xr,xi)
        # xr, xi = x.real, x.imag
        outputs_xr = xr.view(xr.size(0), -1)
        outputs_xi = xi.view(xi.size(0), -1)
        # outputs = torch.complex(outputs_xr,outputs_xi)
        # outputs = outputs.abs()
        # outputs = inputs.view(inputs.size(0), -1)
        return outputs_xr, outputs_xi

    def freeze_weight(self):
        for p in self.parameters():
            p.requires_grad = False
    
class GNN(myModel):
    def __init__(self, cnn_feature_size, gnn_feature_size, nway):
        super(GNN, self).__init__()

        num_inputs = cnn_feature_size + nway
        graph_conv_layer = 2
        self.gnn_obj = GNN_module(nway=nway, input_dim=num_inputs, 
            hidden_dim=gnn_feature_size, 
            num_layers=graph_conv_layer, 
            feature_type='dense')

    def forward(self, inputs):
        logits = self.gnn_obj(inputs).squeeze(-1)

        return logits
      
class gnnModel(myModel):
    def __init__(self, nway):
        super(myModel, self).__init__()
        image_size = 32
        cnn_feature_size = 64
        cnn_hidden_dim = 32
        cnn_num_layers = 3           # 128 对应5层 32对应3层

        gnn_feature_size = 32

        self.cnn_feature = EmbeddingCNN(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
        self.gnn = GNN(cnn_feature_size, gnn_feature_size, nway)

    def forward(self, data):
        [x, _, _, _, xi, _, one_hot_yi, _] = data

        z_xr, z_xi = self.cnn_feature(x)
        z = z_xr + z_xi * 1j
        # zi_s_xr, zi_s_xi = [self.cnn_feature(xi[:, i, :, :, :])[0] for i in range(xi.size(1))]
        zi_s_xr = [self.cnn_feature(xi[:, i, :, :, :])[0] for i in range(xi.size(1))]
        zi_s_xi = [self.cnn_feature(xi[:, i, :, :, :])[1] for i in range(xi.size(1))]
        zi_s_xr = torch.stack(zi_s_xr, dim=1)
        zi_s_xi = torch.stack(zi_s_xi, dim=1)

        zi_s = zi_s_xr + zi_s_xi * 1j
        # zi_s = torch.stack(zi_s, dim=1)

        # follow the paper, concatenate the information of labels to input features
        uniform_pad = torch.FloatTensor(one_hot_yi.size(0), 1, one_hot_yi.size(2)).fill_(
            1.0/one_hot_yi.size(2))
        uniform_pad = torch.complex(uniform_pad,uniform_pad)
        uniform_pad = tensor2cuda(uniform_pad)

        labels = torch.cat([uniform_pad, one_hot_yi], dim=1)
        features = torch.cat([z.unsqueeze(1), zi_s], dim=1)
        # features_xi = torch.cat([z_xi.unsqueeze(1), zi_s_xi], dim=1)

        nodes_features = torch.cat([features, labels], dim=2)

        out_logits = self.gnn(inputs=nodes_features)
        out_logits_mu_real = F.log_softmax(out_logits.real, dim=1)
        out_logits_mu_imag = F.log_softmax(out_logits.imag, dim=1)
        # out_logits_mu = abs(out_logits)
        logsoft_prob = out_logits_mu_real + out_logits_mu_imag * 1j

        return logsoft_prob

class Trainer():
    def __init__(self, trainer_dict):

        self.num_labels = 8

        self.args = trainer_dict['args']
        self.logger = trainer_dict['logger']

        if self.args.todo == 'train':
            self.tr_dataloader = trainer_dict['tr_dataloader']

        if self.args.model_type == 'gnn':
            Model = gnnModel
        
        self.model = Model(nway=self.args.nway)

        self.logger.info(self.model)

        self.total_iter = 0
        self.sample_size = 32

    def load_model(self, model_dir):
        self.model.load(model_dir)

        print('load model sucessfully...')

    def load_pretrain(self, model_dir):
        self.model.cnn_feature.load(model_dir)

        print('load pretrain feature sucessfully...')
    
    def model_cuda(self):
        if torch.cuda.is_available():
            self.model.cuda()

    def eval(self, dataloader, test_sample):
        self.model.eval()
        args = self.args
        iteration = int(test_sample/self.args.batch_size)

        total_loss = 0.0
        total_sample = 0
        total_correct = 0
        total_losst = 0.0
        total_samplet = 0
        total_correctt = 0
        with torch.no_grad():
            for i in range(iteration):
                data = dataloader.load_tr_batch(batch_size=args.batch_size,
                    nway=args.nway, num_shots=args.shots)
                data_test = dataloader.load_te_batch(batch_size=args.batch_size,
                                                nway=args.nway, num_shots=args.shots)

                data_cuda = [tensor2cuda(_data) for _data in data]
                data_cuda_test = [tensor2cuda(_data) for _data in data_test]

                logsoft_prob = self.model(data_cuda)

                label = data_cuda[1]
                # loss = F.nll_loss(logsoft_prob, label)
                loss_real = F.nll_loss(logsoft_prob.real, label)
                loss_imag = F.nll_loss(logsoft_prob.imag, label)
                loss = torch.sqrt(torch.pow(loss_real, 2) + torch.pow(loss_imag, 2))
                # loss = F.nll_loss(logsoft_prob, label)

                total_loss += loss.item() * logsoft_prob.shape[0]

                logsoft_prob_pred = logsoft_prob.real+logsoft_prob.imag
                pred = torch.argmax(logsoft_prob_pred, dim=1)

                # print(pred)

                # print(torch.eq(pred, label).float().sum().item())
                # print(label)

                assert pred.shape == label.shape

                total_correct += torch.eq(pred, label).float().sum().item()
                total_sample += pred.shape[0]

                logsoft_probt = self.model(data_cuda_test)

                labelt = data_cuda_test[1]
                # loss = F.nll_loss(logsoft_prob, label)
                loss_realt = F.nll_loss(logsoft_probt.real, label)
                loss_imagt = F.nll_loss(logsoft_probt.imag, label)
                losst = torch.sqrt(torch.pow(loss_realt, 2) + torch.pow(loss_imagt, 2))
                # loss = F.nll_loss(logsoft_prob, label)

                total_losst += losst.item() * logsoft_probt.shape[0]

                logsoft_prob_predt = logsoft_probt.real + logsoft_probt.imag
                predt = torch.argmax(logsoft_prob_predt, dim=1)

                # print(pred)

                # print(torch.eq(pred, label).float().sum().item())
                # print(label)

                assert predt.shape == labelt.shape

                total_correctt += torch.eq(predt, labelt).float().sum().item()
                total_samplet += predt.shape[0]
        # print('correct: %d / %d' % (total_correct, total_sample))
        # print('correct: %d / %d' % (total_correctt, total_samplet))
        # print(total_correct)
        print('va loss: %.5f, va acc: %.4f %%' % (total_losst / total_samplet, 100.0 * total_correctt / total_samplet))
        return total_loss / total_sample, 100.0 * total_correct / total_sample, 100.0 * total_correctt / total_samplet

    def train_batch(self,i):
        self.model.train()
        args = self.args

        data = self.tr_dataloader.load_tr_batch(batch_size=args.batch_size, 
            nway=args.nway, num_shots=args.shots)

        data_cuda = [tensor2cuda(_data) for _data in data]
        # summary(self.model, input_size=(1, 32, 32), data = data_cuda)
        self.opt.zero_grad()

        logsoft_prob = self.model(data_cuda)

        # print('pred', torch.argmax(logsoft_prob, dim=1))
        # print('label', data[2])
        label = data_cuda[1]
        loss_real = F.nll_loss(logsoft_prob.real, label)
        loss_imag = F.nll_loss(logsoft_prob.imag, label)
        loss = torch.sqrt(torch.pow(loss_real, 2) + torch.pow(loss_imag, 2))
        # loss = loss_real + loss_imag
        # loss = F.nll_loss(logsoft_prob, label)
        loss.backward()
        self.opt.step()
        # print("第%d个iter的学习率：%f" % (i, self.opt.param_groups[0]['lr']))
        self.scheduler_1.step()

        return loss.item()

    def train(self):
        if self.args.freeze_cnn:
            self.model.cnn_feature.freeze_weight()
            print('freeze cnn weight...')

        best_loss = 1e8
        best_acc = 0.0
        stop = 0
        eval_sample = 64
        self.model_cuda()
        self.model_dir = os.path.join(self.args.model_folder, 'model.pth')

        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr,
            weight_decay=1e-6)
        L = range(10000)
        self.scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=list(L[500:3000:500]), gamma=0.5)
        # self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, 
        #     weight_decay=1e-6)

        start = time()
        tr_loss_list = []
        for i in range(self.args.max_iteration):

            # print(i)
            tr_loss = self.train_batch(i)
            tr_loss_list.append(tr_loss)

            if i % self.args.log_interval == 0:
                self.logger.info('iter: %d, spent: %.4f s, tr loss: %.5f' % (i, time() - start, 
                    np.mean(tr_loss_list)))
                del tr_loss_list[:]
                start = time()  

            if i % self.args.eval_interval == 0:
                va_loss, va_acc, val_acc = self.eval(self.tr_dataloader, eval_sample)

                self.logger.info('================== eval ==================')
                self.logger.info('iter: %d, va loss: %.5f, va acc: %.4f %%' % (i, va_loss, va_acc))
                self.logger.info('==========================================')

                if va_loss < best_loss:
                    stop = 0
                    best_loss = va_loss
                    best_acc = va_acc
                    if self.args.save:
                        self.model.save(self.model_dir)

                stop += 1
                start = time()
            
                if stop > self.args.early_stop:
                    break

                if val_acc > 100:
                    if self.args.save:
                        self.model.save(self.model_dir)
                    break

            self.total_iter += 1

        self.logger.info('============= best result ===============')
        self.logger.info('best loss: %.5f, best acc: %.4f %%' % (best_loss, best_acc))

    def test(self, test_data_array, te_dataloader):
        self.model_cuda()
        self.model.eval()
        start = 0
        end = 0
        args = self.args
        batch_size = args.batch_size
        pred_list = []

        with torch.no_grad():
            while start < test_data_array.shape[0]:
                end = start + batch_size 
                if end >= test_data_array.shape[0]:
                    batch_size = test_data_array.shape[0] - start

                data = te_dataloader.load_te_batch(batch_size=batch_size, nway=args.nway, 
                    num_shots=args.shots)

                test_x = test_data_array[start:end]

                data[0] = np2cuda(test_x)

                data_cuda = [tensor2cuda(_data) for _data in data]

                map_label2class = data[-1].cpu().numpy()

                logsoft_prob = self.model(data_cuda)
                # print(logsoft_prob)
                logsoft_prob_pred = logsoft_prob.real + logsoft_prob.imag
                pred = torch.argmax(logsoft_prob_pred, dim=1).cpu().numpy()

                pred = map_label2class[range(len(pred)), pred]

                pred_list.append(pred)

                start = end

        return np.hstack(pred_list)

    def pretrain_eval(self, loader, cnn_feature, classifier):
        total_loss = 0 
        total_sample = 0
        total_correct = 0

        with torch.no_grad():

            for j, (data, label) in enumerate(loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))
                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                total_loss += loss.item() * output.shape[0]

                pred = torch.argmax(output, dim=1)

                assert pred.shape == label.shape

                total_correct += torch.eq(pred, label).float().sum().item()
                total_sample += pred.shape[0]

        return total_loss / total_sample, 100.0 * total_correct / total_sample

    def pretrain(self, pretrain_dataset, test_dataset):
        pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, 
                batch_size=self.args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=self.args.batch_size, shuffle=True)

        self.model_cuda()

        best_loss = 1e8
        self.model_dir = os.path.join(self.args.model_folder, 'pretrain_model.pth')

        cnn_feature = self.model.cnn_feature
        classifier = nn.Linear(list(cnn_feature.parameters())[-3].shape[0], self.num_labels)
        
        if torch.cuda.is_available():
            classifier.cuda()
        self.pretrain_opt =  torch.optim.Adam(
            list(cnn_feature.parameters()) + list(classifier.parameters()), 
            lr=self.args.lr, 
            weight_decay=1e-6)

        start = time()

        for i in range(10000):
            total_tr_loss = []
            for j, (data, label) in enumerate(pretrain_loader):
                data = tensor2cuda(data)
                label = tensor2cuda(label)
                output = classifier(cnn_feature(data))

                output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, label)

                self.pretrain_opt.zero_grad()
                loss.backward()
                self.pretrain_opt.step()
                total_tr_loss.append(loss.item())

            te_loss, te_acc = self.pretrain_eval(test_loader, cnn_feature, classifier)
            self.logger.info('iter: %d, tr loss: %.5f, spent: %.4f s' % (i, np.mean(total_tr_loss), 
                time() - start))
            self.logger.info('--> eval: te loss: %.5f, te acc: %.4f %%' % (te_loss, te_acc))

            if te_loss < best_loss:
                stop = 0
                best_loss = te_loss
                if self.args.save:
                    cnn_feature.save(self.model_dir)

            stop += 1
            start = time()
        
            if stop > self.args.early_stop_pretrain:
                break



if __name__ == '__main__':
    import os
    b_s = 10
    nway = 5
    shots = 5
    batch_x = torch.rand(b_s, 3, 32, 32).cuda()
    batches_xi = [torch.rand(b_s, 3, 32, 32).cuda() for i in range(nway*shots)]

    label_x = torch.rand(b_s, nway).cuda()

    labels_yi = [torch.rand(b_s, nway).cuda() for i in range(nway*shots)]

    print('create model...')
    model = gnnModel(128, nway).cuda()
    # print(list(model.cnn_feature.parameters())[-3].shape)
    # print(len(list(model.parameters())))
    print(model([batch_x, label_x, None, None, batches_xi, labels_yi, None]).shape)
