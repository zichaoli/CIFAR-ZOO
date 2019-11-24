# -*- coding: utf-8 -*-
#_author_='zichao';
#date: 11/13/19 9:52
import torch
import torch.nn as nn
from models.resnet import BasicBlock

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

class Conv1(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride, padding):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False).cuda()
        self.eps = torch.FloatTensor([0.01]).cuda()

    def forward(self, input, grad):
        adv = self.eps * grad.sign()
        new_input = input + adv
        new_output = self.conv(new_input)

        return new_output

# class Small_group(nn.Module):
#     def __init__(self, in_planes, planes, kernel_size, stride, padding):
#         super(Small_group, self).__init__()
#         self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False).cuda()
#         self.eps = torch.FloatTensor([0.01]).cuda()
#         self.bn =

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicGroup(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicGroup, self).__init__()
        self.conv_1 = conv3x3(inplanes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.eps = torch.FloatTensor([0.0001]).cuda()

    def forward(self, x, grad):
        residual = x
        x = x + self.eps * grad.sign()
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)

        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out

class Layer_loss(nn.Module):
    def __init__(self):
        super(Layer_loss, self).__init__()
    def forward(self, ori_output, new_output, grad):
        loss = (new_output - ori_output).view(-1).dot(grad.contiguous().view(-1))
        return loss


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

            layer = layers[index]
            layer.conv.load_state_dict(p.state_dict())
            layer.zero_grad()
            layer_input = layer_inputs[index]
            layer_output = layer_outputs[index]
            grad_input = grad_inputs[32- index]
            grad_output = grad_outputs[32 - index]
            new_output = layer(layer_input, grad_input)
            # crit = Layer_loss()
            loss = crit(layer_output, new_output, grad_output)
            # loss.backward()
            layer_loss +=loss
            index += 1
            # if index > 0:
    loss_index = 0
    layer_loss = layer_loss/(index + 1)
    layer_loss.backward()
    for p in model.modules():
        if isinstance(p, nn.Conv2d):
            layer =layers[loss_index]
            p.weight.grad = lam * p.weight.grad + (1-lam) * layer.conv.weight.grad
            loss_index += 1

    return layer_loss

def group_noise(model, groups, crit, alpha):
    index = 0
    layer_loss = 0
    for p in model.modules():
        if isinstance(p, BasicBlock):
            group = groups[index]
            group.load_state_dict(p.state_dict())
            group.zero_grad()
            # if index > 0 and index%5 == 0 :
            #     group.eps *= mults
            group_input = p.info['input']
            group_output = p.info['output']
            grad_input = p.info['input_grad']
            grad_output = p.info['output_grad']
            new_output = group(group_input, grad_input)
            loss = crit(group_output, new_output, grad_output)

            loss = loss/len(groups)
            loss.backward()

            # for para in p.parameters():
            #     print('yes')
            p.conv_1.weight.grad = alpha * p.conv_1.weight.grad + (1 - alpha) * group.conv_1.weight.grad
            p.conv_2.weight.grad = alpha * p.conv_2.weight.grad + (1 - alpha) * group.conv_2.weight.grad

            layer_loss += loss.item()
            index +=1

    return layer_loss
