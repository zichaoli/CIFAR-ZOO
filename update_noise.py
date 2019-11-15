# -*- coding: utf-8 -*-
#_author_='zichao';
#date: 11/13/19 9:52
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False).cuda()
        self.eps = torch.FloatTensor([0.001]).cuda()

    def forward(self, input, grad):
        adv = self.eps * grad.sign()
        new_input = input + adv
        new_output = self.conv(new_input)

        return new_output

class Layer_loss(nn.Module):
    def __init__(self):
        super(Layer_loss, self).__init__()
    def forward(self, ori_output, new_output, grad):
        loss = (new_output - ori_output).view(-1).dot(grad.contiguous().view(-1))
        return loss

result = []
def hook(module, input, output):
    result.append(input)
    result.append(output)

def backward_hook(module, grad_input, grad_output):
    pass

def update_grad(model, layer_inputs, layer_outputs, grad_inputs, grad_outputs, layers, crit, lam):
    index = 0
    layer_loss = 0
    for p in model.modules():
        if isinstance(p, nn.Conv2d):
            # in_planes = p.in_channels
            # planes = p.out_channels
            # kernel_size = p.kernel_size[0]
            # padding = p.padding[0]
            # stride = p.stride[0]
            #
            # layer = Conv(in_planes, planes, kernel_size, stride,padding)
            # layer.conv.load_state_dict(p.state_dict())
            layer = layers[index]
            layer.zero_grad()
            layer_input = layer_inputs[index]
            layer_output = layer_outputs[index]
            grad_input = grad_inputs[20 - index]
            grad_output = grad_outputs[20 - index]
            new_output = layer(layer_input, grad_input)
            # crit = Layer_loss()
            loss = crit(layer_output, new_output, grad_output)
            loss.backward()
            layer_loss +=(1-lam)*loss

            p.weight.grad = lam * p.weight.grad + (1-lam) * layer.conv.weight.grad

            index += 1
    return layer_loss