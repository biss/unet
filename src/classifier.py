
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import FloatTensor as FT
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.nn as nn


# we can play around with dilation variable with is currently set to 1
def make_conv_layer_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        # the bias term is not need because batch normalisation already takes care of that
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        ]

def make_conv_layer_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.ReLu(inplace=True)
        ]


class UNet256_3x3(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_3x3, self).__init__()
        in_channels, height, width = in_shape

        self.down1 = nn.Sequential(
                *make_conv_layer_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(16, 32, kernel_size=3, stride=2, padding=1)
                )

        self.down2 = nn.Sequential(
                *make_conv_layer_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(64, 128, kernel_size=3, stride=1, padding=1)
                )

        self.down3 = nn.Sequential(
                *make_conv_layer_bn_relu(128, 256, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(256, 512, kernel_size=3, stride=1, padding=1)
                )

        self.down4 = nn.Sequential(
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.same = nn.Sequential(
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.up4 = nn.Sequential(
                *make_conv_layer_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 512, kernel_size=3, stride=1, padding=1)
                )

        self.up3 = nn.Sequential(
                *make_conv_layer_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(512, 128, kernel_size=3, stride=1, padding=1)
                )

        self.up2 = nn.Sequential(
                *make_conv_layer_bn_relu(256, 128, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(128, 32, kernel_size=3, stride=1, padding=1)
                )

        self.up1 = nn.Sequential(
                *make_conv_layer_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
                *make_conv_layer_bn_relu(64, 32, kernel_size=3, stride=1, padding=1)
                )

        self.up0 = nn.Sequential(
                *make_conv_layer_bn_relu(32, 32, kernel_size=3, stride=1, padding=1)
                )

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # x is (3, 256, 256)
        down1 = self.down1(x) # down1 is (32, 128, 128)
        out = F.max_pool2d(down1, kernel_size=2, stride=2) # out is (32, 64, 64)

        down2 = self.down2(out) # down2 is (128, 64, 64)
        out = F.max_pool2d(down2, kernel_size=2, stride=2) # out is (128, 32, 32)

        down3 = self.down3(out) # down3 is (512, 32, 32)
        out = F.max_pool2d(down3, kernel_size=2, stride=2) # out is (512, 16, 16)

        down4 = self.down4(out) # down4 is (512, 16, 16)
        out = F.max_pool2d(down4, kernel_size=2, stride=2) # out is (512, 8, 8)

        out = self.same(out) # down4 is (512, 8, 8)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (512, 16, 16)
        out = torch.cat([down4, out], 1) # out is (1024, 16, 16)
        out = self.up4(out) # out is (512, 16, 16)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (512, 32, 32)
        out = torch.cat([down3, out], 1) # out is (1024, 32, 32)
        out = self.up3(out) # out is (128, 32, 32)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (128, 64, 64)
        out = torch.cat([down2, out], 1)# out is (256, 64, 64)
        out = self.up2(out)# out is (32, 64, 64)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (32, 128, 128)
        out = torch.cat([down1, out], 1) # out is (64, 128, 128)
        out = self.up1(out) # out is (32, 128, 128)

        out = F.upsample(out, scale_factor=2, mode='bilinear') # out is (32, 256, 256)
        out = self.up0(out) # out is (32, 128, 128)

        out = self.classify(out) # out is (3, 128, 128)
        return out

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NucleiClassifier:

    def __init__(self, net, criteria, metric, optimizer, gpu=1):
        self.net = net
        self.criteria = criteria
        self.metric = metric
        self.optimizer = optimizer
        self.use_gpu = torch.cuda.is_available() # if available train on gpu
        self.gpu = gpu # gpu id to use for training
        self.epoch_counter = 0

    def _validate_epoch(self, valid_loader, threshold):
        losses = AverageMeter()
        metrics = AverageMeter()
        batch_size = valid_loader.batch_size
        for id_, (inputs, targets, input_id) in enumerate(valid_loader):
            if self.use_gpu:
                inputs = inputs.cuda(self.gpu)
                targets = targets.cuda(self.gpu)
            inputs, targets = V(inputs, volatile=True), V(targets, volatile=True)

            # forward prop
            logits = self.net(inputs)
            loss = self.criteria(logits, targets)
            losses.update(loss.data[0], batch_size)
            metric = self.metric(logits, targets, threshold)
            metrics.update(metric.data[0], batch_size)
        return losses.avg, metrics.avg

    def _train_epoch(self, train_loader, threshold):
        losses = AverageMeter()
        metrics = AverageMeter()
        batch_size = train_loader.batch_size
        for id_, (inputs, targets, input_id) in enumerate(train_loader):
            if self.use_gpu:
                inputs = inputs.cuda(self.gpu)
                targets = targets.cuda(self.gpu)
            inputs, targets = V(inputs, volatile=True), V(targets, volatile=True)

            # forward prop
            logits = self.net(inputs)

            # record losses and metrics
            loss = self.criteria(logits, targets)
            losses.update(loss.data[0], batch_size)
            metric = self.metric(logits, targets, threshold)
            metrics.update(metric.data[0], batch_size)

            # back propagate - adjust weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses.avg, metrics.avg


    def _run_epoch(self, train_loader, valid_loader, threshold):
        """
        Run a single epoch print training and validation loss
        """
        # set model in train mode and run a train pass
        self.net.train()
        train_loss, train_metric = self._train_epoch(train_loader, threshold)

        # set model in eval mode and validate epoch
        self.net.eval()
        val_loss, val_metric = self._validate_epoch(valid_loader, threshold)
        self.epoch_counter += 1

        print("Epoch: {}".format(self.epoch_counter))
        print("LOSS - Training : [{}], Validation : [{}]".format(round(train_loss, 4), round(val_loss, 4)))
        print("METRIC - Training : [{}], Validation : [{}]".format(round(train_metric, 4), round(val_metric, 4)))
        return val_loss, val_metric


    def train(self, train_loader, valid_loader, epochs, threshold=0.5):
        """
        Train model for n epochs
        Inputs:
            train_loader (Dataloader): PyTorch dataloader for training
            valid_loader (Dataloader): PyTorch dataloader for validation
            epochs (int): Number of epochs to train the model for
            threshold (float): Threshold to use for evaluation metrics
        """
        if self.use_gpu:
            self.net.cuda(self.gpu)

        for epoch in range(epochs):

            loss, acc = self._run_epoch(train_loader, valid_loader, threshold)
            best_accuracy = acc

            # Get bool not ByteTensor
            is_best = bool(acc > best_accuracy)
            # Get greater Tensor to keep track best acc
            best_accuracy = max(acc, best_accuracy)
            # Save checkpoint if is a new best
            save_checkpoint({
                'epoch': self.epoch_counter,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy
            }, is_best)


    def save_model(self, path="/model"):
        """
        Save model parameters to the given path
        Inputs:
            model_path (str): The path to the model to restore
        """
        state = {
            'epoch': self.epoch_counter,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)

    def restore_model(self, path):
        """
        Restore a model parameters from the given path
        Inputs:
            path (str): The path to the model to restore
            gpu (int): GPU instance to continue training
        """
        # if cuda is not available load everything to cpu
        if not self.use_cuda:
            state = torch.load(path, map_location=lambda storage, loc: storage)
        else:
            state = torch.load(path)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch_counter = state['epoch'] # counts number of epochs

    def predict(self, test_loader):
        # eval mode
        self.net.eval()
        preds = []
        for ind, (images, index) in enumerate(test_loader):
            if self.use_cuda:
                images = images.cuda(self.gpu)

            images = V(images, volatile=True)

            # forward
            logits = self.net(images)
            probs = F.sigmoid(logits)
            probs = probs.data.cpu().numpy()
            preds.append(probs)
        return preds

    def save_checkpoint(state, is_best, filename='checkpoint/chpt.tar'):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print ("=> Saving a new best")
            torch.save(state, filename)  # save checkpoint
        else:
            print ("=> Validation Accuracy did not improve")

    def resume_checkpoint(self, in_shape, num_classes,
                          resume_weights='checkpoint/chpt.tar'):
        # cuda = torch.cuda.is_available()
        if cuda:
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights,
                                    map_location=lambda storage,
                                    loc: storage)
        model = UNet256_3x3(in_shape, num_classes)
        epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(
                resume_weights, checkpoint['epoch']))
        self.net = model
