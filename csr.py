import torch.nn
from torch.optim import lr_scheduler

import more.train
import utils.arguments
import utils.parm
import utils.plot

from more.datasets import load_more_data
from more.network import MoreNet
from utils.centerloss import CenterLoss
from torch.utils.tensorboard import SummaryWriter

utils.plot.fixLiberror()
options = utils.arguments.getCmdArgs()

writer = SummaryWriter(options['log_dir'])

options['num_classes'] = 8
options['name'] = 'close'
train_loader, test_loader = load_more_data(options['data_dir'], options['batch_size'])

feat_dim = 2
options['feat_dim'] = feat_dim

net = MoreNet().to(options['device'])
loss = torch.nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=options['num_classes'], feat_dim=2, use_gpu=options['use_gpu'])
optimizer_model = torch.optim.SGD(net.parameters(), lr=options['lr'], weight_decay=5e-04, momentum=0.9)
optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)

scheduler = lr_scheduler.StepLR(optimizer_model, step_size=20, gamma=0.5)

utils.parm.init_net(net, options)
# utils.parm.init_parm(criterion_cent,"center",options)

if options['eval']:
    exit(0)

#more.train.train_center_dry(net, train_loader, options['use_gpu'], options['num_classes'], options['name'], -1)
for epoch in range(options['max_epoch']):
    print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
    # more.train.train_center(net, loss, criterion_cent, optimizer_model, optimizer_centloss, train_loader, options['use_gpu'], options['num_classes'],options['name'], epoch)
    more.train.train_ch6(net, loss, optimizer_model, train_loader, test_loader, epoch, "cuda:0", writer)

    scheduler.step()
    if epoch % options['eval_freq'] == 0:
        utils.parm.save_net(net, options)
        # utils.parm.save(criterion_cent, "center", options)
        pass
