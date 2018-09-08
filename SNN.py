import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SNNLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.w = nn.Parameter(torch.rand(output_size,input_size)*(5./input_size)+(1./input_size))
        print(self.w)

    def forward(self, input):
        batch_size = input.size()[0]
        sorted_input, sorted_indices = input.sort(dim=1)
        weight_outsize_not_sorted = self.w.repeat(batch_size,1).view(batch_size,self.output_size,self.input_size)
        indice_outsize = sorted_indices.repeat(1,1,self.output_size).view(batch_size,self.output_size,self.input_size)
        weight_outsize = torch.gather(weight_outsize_not_sorted,2,indice_outsize)
        input_outsize = sorted_input.repeat(1,1,self.output_size).view(batch_size,self.output_size,self.input_size)
        weight_input_mul = input_outsize * weight_outsize
        for index in range(1,self.input_size,1):
            weight_input_mul[:,:,index] += weight_input_mul[:,:,index-1]
            weight_outsize[:,:,index] += weight_outsize[:,:,index-1]
        out_all = weight_input_mul / torch.clamp(weight_outsize - 1, 1e-10, 1e10)
        return out_all



l = SNNLayer(3,4)
list = [[1.,1.,1.],[1.,1.,1.],[2.,1.,5.],[9.,3.,4.],[2.,5.,9.]]
layerin = torch.tensor(list)
layerout = l(layerin)
print(layerout)
