from utils import AverageMeter
from utils.plot import plot_features
import numpy as np


def train_center(model, criterion_xent, criterion_cent,
                 optimizer_model, optimizer_centloss,
                 trainloader, use_gpu, num_classes, epoch):
    weight_cent = 1
    print_freq=50

    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data,True)
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))


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
    plot_features(all_features, all_labels, num_classes, epoch, prefix='more',legend=trainloader.dataset.get_labels_name())
