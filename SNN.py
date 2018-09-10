import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dtypeFloat = torch.cuda.FloatTensor
dtypeInt = torch.cuda.LongTensor
dtypeByte = torch.cuda.ByteTensor

class SNNLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.w = nn.Parameter(torch.rand(output_size,input_size).type(dtypeFloat)*(10./input_size)+(0./input_size))
        print(self.w)

    def forward(self, input):
        batch_size = input.size()[0]
        sorted_input, sorted_indices = input.sort(dim=1)
        sorted_input.type(dtypeFloat)
        sorted_indices.type(dtypeInt)
        weight_outsize_not_sorted = self.w.repeat(batch_size,1).view(batch_size,self.output_size,self.input_size).type(dtypeFloat)
        indice_outsize = sorted_indices.repeat(1,1,self.output_size).view(batch_size,self.output_size,self.input_size).type(dtypeInt)
        weight_outsize = torch.gather(weight_outsize_not_sorted,2,indice_outsize).type(dtypeFloat)
        input_outsize = sorted_input.repeat(1,1,self.output_size).view(batch_size,self.output_size,self.input_size).type(dtypeFloat)
        weight_input_mul = (input_outsize * weight_outsize).type(dtypeFloat)
        weight_input_mul_sum = weight_input_mul[:,:,0].view(batch_size,self.output_size,1)
        weight_outsize_sum = weight_outsize[:,:,0].view(batch_size,self.output_size,1)
        for index in range(1,self.input_size,1):
            weight_input_mul_sum = torch.cat((weight_input_mul_sum, weight_input_mul[:,:,index].view(batch_size,self.output_size,1) + weight_input_mul[:,:,index-1].view(batch_size,self.output_size,1)), 2)
            weight_outsize_sum = torch.cat((weight_outsize_sum,weight_outsize[:,:,index].view(batch_size,self.output_size,1) + weight_outsize[:,:,index-1].view(batch_size,self.output_size,1)), 2)
        print(weight_input_mul_sum)
        out_all = weight_input_mul_sum / torch.clamp(weight_outsize_sum - 1, 1e-10, 1e10)
        out_all = torch.cat((out_all,1e10*torch.ones([batch_size,self.output_size,1]).type(dtypeFloat)),2)
        input_outsize = torch.cat((input_outsize,torch.zeros([batch_size,self.output_size,1]).type(dtypeFloat)),2)
        out_cond1 = out_all > input_outsize
        out_cond2 = weight_outsize_sum > 1.
        out_cond2 = torch.cat((out_cond2,torch.ones([batch_size,self.output_size,1]).type(dtypeByte)),2)
        out_cond = out_cond1 * out_cond2
        #print(out_cond)
        _, index = torch.max(out_cond, 2)
        return torch.gather(out_all,2,index.view(batch_size,self.output_size,1)).squeeze()


class LossModule(nn.Module):
    def __init__(self):
        super(LossModule, self).__init__()

    def forward(self, input):
        [output,groundtruth] = input
        output = 0 - output
        exp_out = torch.exp(output).type(dtypeFloat)
        exp_real = exp_out * groundtruth.type(dtypeFloat)
        sum_exp = torch.sum(exp_out,dim=1)
        sum_real = torch.sum(exp_real,dim=1)
        return torch.mean(0 - torch.log(torch.clamp(sum_real / torch.clamp(sum_exp,1e-10,1e10),1e-10,1)))


class WeightSumCost(nn.Module):
    def __init__(self,K):
        super(WeightSumCost, self).__init__()
        self.K = K

    def forward(self, input):
        input.type(dtypeFloat)
        step1 = 1 - input
        cond = step1 > 0
        step2 = step1*cond.type(dtypeFloat)
        sum = torch.sum(step2)
        return self.K * sum


class L2Cost(nn.Module):
    def __init__(self,K):
        super(L2Cost, self).__init__()

    def forward(self, input):
        sqr = torch.mul(input.type(dtypeFloat),input.type(dtypeFloat))
        return torch.sum(sqr)

l = SNNLayer(784,800)
c = LossModule()
weightsc = WeightSumCost(1)
l2c = L2Cost(1)
list = np.random.rand(10,784)
gt = np.zeros((10,800))
i=0
while True:
    layerin = torch.tensor(list)
    groundt = torch.tensor(gt)
    layerout = l(layerin)
    print(layerout)
    loss = c([layerout,groundt])
    wsc = weightsc(l.w)
    l2 = l2c(l.w)
    cost = loss + wsc + l2
    cost.backward()
    print(l.w.grad)
    l.w.grad.zero_()
    print(i)
    i+=1
print(layerout)
