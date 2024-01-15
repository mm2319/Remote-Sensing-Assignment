import numpy as np
import torch
from tqdm import tqdm
from . import metrics
import torch.nn.functional as F
from .losses import *

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def metric(input, target):
    """
    Args:
        input (tensor): prediction
        target (tensor): reference data

    Returns:
        float: harmonic fscore without including backgorund
    """
    input = torch.softmax(input, dim=1)
    scores = []

    for i in range(1,input.shape[1]):  # background is not included
        ypr = input[:, i, :, :].view(input.shape[0], -1)
        ygt = target[:, i, :, :].view(target.shape[0], -1)
        scores.append(metrics.iou(ypr, ygt).item())

    return np.mean(scores)


def valid_multi_loss(model=None, dataloader=None, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.eval().to(device)
    loss_1 = CEWithLogitsLoss()
    loss_2 = FocalLoss()
    loss_3 = JaccardLoss()
    loss_4 = MCCLoss()
    loss_5 = DiceLoss()
    iterator = tqdm(dataloader, desc="Valid")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)

            loss = loss_5(outputs, y) + loss_1(outputs, y) 

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def train_multi_loss(model, optimizer, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    loss_1 = JaccardLoss()
    loss_2 = CEWithLogitsLoss()
    loss_3 = FocalLoss()
    loss_4 = MCCLoss()
    loss_5 = DiceLoss()
    logs = {}

    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)


        loss = loss_2(outputs, y) + loss_5(outputs, y)#loss.detach_().requires_grad_(True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def train_epoch(model, optimizer, criterion, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.train().to(device)

    iterator = tqdm(dataloader, desc="Train")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)
        outputs = F.softmax(outputs, dim=1)
        label = torch.argmax(y, dim=1)

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(outputs, y), n=n)
      
        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs

def train(model, optimizer, criterion, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)


        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs

def valid_epoch(model=None, criterion=None, dataloader=None, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.eval().to(device)

    iterator = tqdm(dataloader, desc="Valid")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)
            outputs = F.softmax(outputs, dim=1)
            label = torch.argmax(y, dim=1)

            loss = criterion(outputs, label)

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def valid(model=None, criterion=None, dataloader=None, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.eval().to(device)

    iterator = tqdm(dataloader, desc="Valid")
    for x, y, *_ in iterator:
        x = x.to(device).float()
        y = y.to(device).float()
        n = x.shape[0]
       
        with torch.no_grad():
            outputs = model.forward(x)

            loss = criterion(outputs, y)

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs
