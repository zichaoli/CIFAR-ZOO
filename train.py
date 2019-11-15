# -*-coding:utf-8-*-
import argparse
import logging
import yaml
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict
from models import *
from update_noise import Layer_loss, Conv, update_grad

from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work_path', required=True, type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lam', type=float, default=0.9)

args = parser.parse_args()
logger = Logger(log_file_name=args.work_path + '/log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()



def train(train_loader, net, criterion, optimizer, epoch, device,\
          layer_inputs, layer_outputs, grad_inputs, grad_outputs, layers, crit):
    global writer

    start = time.time()
    net.train()



    train_loss = 0
    correct = 0
    total = 0
    eps = 0.001
    lam = 0.5
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True
        layer_inputs.clear()
        layer_outputs.clear()
        grad_inputs.clear()
        grad_outputs.clear()
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()

#fgsm
        # for p in net.parameters():
        #     p.grad *= lam
        # adv_input = inputs + eps * inputs.grad.sign()
        #
        # outputs = net(adv_input)
        #
        # loss_2 = (1-lam) * criterion(outputs, targets)
        # loss_2.backward()
#
        layer_loss = update_grad(net, layer_inputs, layer_outputs, grad_inputs, grad_outputs, layers, crit, args.lam)

        optimizer.step()

        # count the loss and acc
        train_loss += args.lam*loss.item() + layer_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device,\
         layer_inputs, layer_outputs, grad_inputs, grad_outputs):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            layer_inputs.clear()
            layer_outputs.clear()
            grad_inputs.clear()
            grad_outputs.clear()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # read config from yaml file
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)
#smart noise
    layer_inputs = []
    layer_outputs = []
    grad_inputs = []
    grad_outputs = []

    def forward_hook(module, input, output):
        layer_inputs.append(input[0].detach())
        layer_outputs.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        grad_inputs.append(grad_input[0].detach())
        grad_outputs.append(grad_output[0].detach())

    def input_hook(grad):
        grad_inputs.append(grad)

    def output_hook(grad):
        grad_outputs.append(grad)

    for p in net.modules():
        if isinstance(p, nn.Conv2d):
            p.register_forward_hook(forward_hook)
            p.register_backward_hook(backward_hook)
    layers = []
    for p in net.modules():
        if isinstance(p, nn.Conv2d):
            in_planes = p.in_channels
            planes = p.out_channels
            kernel_size = p.kernel_size[0]
            padding = p.padding[0]
            stride = p.stride[0]

            layer = Conv(in_planes, planes, kernel_size, stride, padding)
            layers.append(layer)
    crit = Layer_loss()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(
        data_augmentation(config))

    transform_test = transforms.Compose(
        data_augmentation(config, is_train=False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        lr = adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device,\
              layer_inputs, layer_outputs, grad_inputs, grad_outputs, layers, crit)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device,\
                 layer_inputs, layer_outputs, grad_inputs, grad_outputs)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    main()
