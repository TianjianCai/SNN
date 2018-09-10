import numpy as np
import os
import torch
import torch.nn as nn

np.set_printoptions(threshold=np.inf)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

dtypeFloat = torch.cuda.FloatTensor
dtypeInt = torch.cuda.LongTensor
dtypeByte = torch.cuda.ByteTensor


class MnistData(object):
    def __init__(
        self,
        size,
        path=[
            "/save/train_data_x",
            "/save/train_data_y"]):
        try:
            self.xs_full = np.load(os.getcwd() + path[0] + ".npy")
            self.ys_full = np.load(os.getcwd() + path[1] + ".npy")
            print(path[0] + ", " + path[1] + " " + "loaded")
        except BaseException:
            self.xs_full, self.ys_full = mnist.train.next_batch(
                size, shuffle=False)
            np.save(os.getcwd() + path[0], self.xs_full)
            np.save(os.getcwd() + path[1], self.ys_full)
            print("cannot load " + path[0] + ", " + path[1] + ", get new data")
        self.datasize = size
        self.pointer = 0

    def next_batch(self, batch_size):
        if self.pointer + batch_size < self.datasize:
            pass
        else:
            self.pointer = 0
            if batch_size >= self.datasize:
                batch_size = self.datasize - 1
        xs = self.xs_full[self.pointer:self.pointer + batch_size, :]
        ys = self.ys_full[self.pointer:self.pointer + batch_size, :]
        self.pointer = self.pointer + batch_size
        return xs, ys


class SNNLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.w = nn.Parameter(torch.rand(output_size, input_size).type(
            dtypeFloat) * (3. / input_size) + (1. / input_size))

    def forward(self, input):
        input = torch.exp(input * 1.79)
        batch_size = input.size()[0]
        sorted_input, sorted_indices = input.sort(dim=1)
        sorted_input.type(dtypeFloat)
        sorted_indices.type(dtypeInt)
        weight_outsize_not_sorted = self.w.repeat(batch_size, 1).view(
            batch_size, self.output_size, self.input_size).type(dtypeFloat)
        indice_outsize = sorted_indices.repeat(1, 1, self.output_size).view(
            batch_size, self.output_size, self.input_size).type(dtypeInt)
        weight_outsize = torch.gather(
            weight_outsize_not_sorted,
            2,
            indice_outsize).type(dtypeFloat)
        input_outsize = sorted_input.repeat(1, 1, self.output_size).view(
            batch_size, self.output_size, self.input_size).type(dtypeFloat)
        weight_input_mul = (input_outsize * weight_outsize).type(dtypeFloat)
        weight_input_mul_sum = weight_input_mul[:, :, 0].view(
            batch_size, self.output_size, 1)
        weight_outsize_sum = weight_outsize[:, :, 0].view(
            batch_size, self.output_size, 1)
        for index in range(1, self.input_size, 1):
            weight_input_mul_sum = torch.cat((weight_input_mul_sum, weight_input_mul[:, :, index].view(
                batch_size, self.output_size, 1) + weight_input_mul[:, :, index - 1].view(batch_size, self.output_size, 1)), 2)
            weight_outsize_sum = torch.cat((weight_outsize_sum, weight_outsize[:, :, index].view(
                batch_size, self.output_size, 1) + weight_outsize[:, :, index - 1].view(batch_size, self.output_size, 1)), 2)
        out_all = weight_input_mul_sum / \
            torch.clamp(weight_outsize_sum - 1, 1e-10, 1e10)
        out_all = torch.cat(
            (out_all, 1e10 * torch.ones([batch_size, self.output_size, 1]).type(dtypeFloat)), 2)
        input_outsize = torch.cat(((torch.ones([batch_size, self.output_size, 2]) * np.inf).type(
            dtypeFloat), input_outsize), 2)[:, :, 1:self.input_size + 2]
        out_cond1 = out_all < input_outsize
        out_cond2 = weight_outsize_sum > 1.
        out_cond2 = torch.cat((out_cond2, torch.ones(
            [batch_size, self.output_size, 1]).type(dtypeByte)), 2)
        out_cond = out_cond1 * out_cond2
        # print(out_cond)
        _, index = torch.max(out_cond, 2)
        return torch.gather(
            out_all,
            2,
            index.view(
                batch_size,
                self.output_size,
                1)).squeeze()


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()

    def forward(self, input):
        [output, groundtruth] = input
        output = 0 - output
        exp_out = torch.exp(output).type(dtypeFloat)
        exp_real = exp_out * groundtruth.type(dtypeFloat)
        sum_exp = torch.sum(exp_out, dim=1)
        sum_real = torch.sum(exp_real, dim=1)
        return torch.mean(
            0 -
            torch.log(
                torch.clamp(
                    sum_real /
                    torch.clamp(
                        sum_exp,
                        1e-10,
                        1e10),
                    1e-10,
                    1)))


class WeightSumCost(nn.Module):
    def __init__(self, K):
        super(WeightSumCost, self).__init__()
        self.K = K

    def forward(self, input):
        input.type(dtypeFloat)
        step1 = 1 - input
        cond = step1 > 0
        step2 = step1 * cond.type(dtypeFloat)
        sum = torch.sum(step2)
        return self.K * sum


class L2Cost(nn.Module):
    def __init__(self, K):
        super(L2Cost, self).__init__()
        self.K = K

    def forward(self, input):
        sqr = torch.mul(input.type(dtypeFloat), input.type(dtypeFloat))
        return self.K * torch.sum(sqr)


def cal_lr(lr, step_num):
    bias = 1e-4
    return (lr * np.exp(step_num * -1e-4)) + bias


def backwardhook(self, grad_input, grad_output):
    print(grad_input)
    print(grad_output)



TRAINING_DATA_SIZE = 50000
TRAINING_BATCH = 10
K1 = 100
K2 = 0.001
learning_rate = 1e0

l1 = SNNLayer(784, 800)
l2 = SNNLayer(800, 10)
l1.register_backward_hook(backwardhook)
loss = LossModule()
weightsc = WeightSumCost(K1)
l2c = L2Cost(K2)

mnistData_train = MnistData(
    size=TRAINING_DATA_SIZE,
    path=[
        "/save/train_data_x",
        "/save/train_data_y"])


i = 0
while True:
    print(i)

    xs, ys = mnistData_train.next_batch(TRAINING_BATCH)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    layer1 = l1(xs)
    layer2 = l2(layer1)
    l = loss([layer2, ys])
    wsc1 = weightsc(l1.w)
    wsc2 = weightsc(l2.w)
    l2c1 = l2c(l1.w)
    l2c2 = l2c(l2.w)
    cost = l + wsc1 + wsc2 + l2c1 + l2c2
    cost.backward()
    print(cost)
    with torch.no_grad():
        l1_g = l1.w.grad
        l2_g = l2.w.grad

        g_sum_sqr = torch.clamp(
            torch.sum(
                torch.mul(
                    l1_g,
                    l1_g)) +
            torch.sum(
                torch.mul(
                    l2_g,
                    l2_g)),
            1e-10,
            1e10)
        l1_g = l1_g / g_sum_sqr
        l2_g = l2_g / g_sum_sqr
        l1.w -= l1_g * cal_lr(learning_rate, i)
        l2.w -= l2_g * cal_lr(learning_rate, i)
        l1.w.grad.zero_()
        l2.w.grad.zero_()
    i += 1
