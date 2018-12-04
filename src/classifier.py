
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import FloatTensor as FT
import torch.nn.functional as F
from torch.autograd import Variable as V


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0

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
        print("LOSS - Training : [{}], Validation : [{}]".format(train_loss, val_loss))
        print("METRIC - Training : [{}], Validation : [{}]".format(train_metric, val_metric))
        return valid_loss, valid_metric

    def train(self, train_loader, valid_loader, epochs, save_file=None, threshold=0.5):
        """
        Train model for n epochs
        Inputs:
            train_loader (Dataloader): PyTorch dataloader for training
            valid_loader (Dataloader): PyTorch dataloader for validation
            epochs (int): Number of epochs to train the model for
            threshold (float): Threshold to use for evaluation metrics
        """
        c_valid_loss = float('Inf')
        if self.use_gpu:
            self.net.cuda(self.gpu)

        for epoch in range(epochs):
            valid_loss, _ = self._run_epoch(train_loader, valid_loader, threshold)
            if valid_loss < c_valid_loss:
                self.save_model(save_file)
                c_valid_loss = valid_loss

    def save_model(self, path="models/default"):
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
        path += str(self.epoch_counter) + ".pth"
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
