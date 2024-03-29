import torch.cuda
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter
from utils.plot import plot_features
import numpy as np
from d2l import torch as d2l


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0
    all_features, all_labels = [], []
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)

            if options['use_gpu']:
                all_features.append(x.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(y.data.numpy())
                all_labels.append(labels.data.numpy())

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg

    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(all_features, all_labels, options['num_classes'], epoch, prefix='arpl',
                  points=criterion.points.data.cpu().numpy(), legends=trainloader.dataset.get_labels_name())
    return loss_all


def train_center(model, criterion_xent, criterion_cent,
                 optimizer_model, optimizer_centloss,
                 trainloader, use_gpu, num_classes, name, epoch):
    weight_cent = 1
    print_freq = 50

    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data, True)
        loss_xent = criterion_xent(outputs, labels)
        # loss_cent = criterion_cent(features, labels)
        # loss_cent *= weight_cent
        # loss = loss_xent + loss_cent
        loss = loss_xent
        optimizer_model.zero_grad()
        # optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        # for param in criterion_cent.parameters():
        #     param.grad.data *= (1. / weight_cent)
        # optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        # cent_losses.update(loss_cent.item(), labels.size(0))

        if use_gpu:
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        else:
            all_features.append(features.data.numpy())
            all_labels.append(labels.data.numpy())

        if (batch_idx + 1) % print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                          cent_losses.val, cent_losses.avg))
    # end of batch loop
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(all_features, all_labels, num_classes, epoch, prefix=name,
                  legends=trainloader.dataset.get_labels_name())


def train_center_dry(model, trainloader, use_gpu, num_classes, name, epoch):
    print_freq = 50

    model.eval()

    all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data, True)
        if use_gpu:
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        else:
            all_features.append(features.data.numpy())
            all_labels.append(labels.data.numpy())
    # end of batch loop
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(all_features, all_labels, num_classes, epoch, prefix=name,
                  legends=trainloader.dataset.get_labels_name())


def train_ch6(net, loss, optimizer, train_iter, test_iter, epoch, device, writer: SummaryWriter):
    print('training on', device)
    net.to(device)
    timer, num_batches = d2l.Timer(), len(train_iter)
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = d2l.Accumulator(3)
    net.train()
    for i, (X, y) in enumerate(train_iter):
        timer.start()
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            writer.add_scalars('Train',{'loss':train_l,'acc':train_acc},epoch)
    test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
    writer.add_scalars('Train', {'test': test_acc}, epoch)
