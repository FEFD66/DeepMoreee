#%%
import time
import datetime

import torch.nn

import utils.arguments
import utils.plot
from more.datasets import load_more_data, load_data_mnist, load_osr_data
from more.network import ARPLNet, init_weights, MoreNet
import more.train as T
from more.ARPLoss import ARPLoss
import os

from more.test import test

utils.plot.fixLiberror()
#%%
options = utils.arguments.getCmdArgs()

use_gpu = torch.cuda.is_available()
if options['cpu']:
    use_gpu = False

if use_gpu:
    print("Using GPU")
else:
    print("Using CPU")
options.update({
    'use_gpu': use_gpu
})

# train_loader, test_loader = load_more_data(options['data_dir'], options['batch_size'])
options['name'] = "osr24507"
known = [3, 5, 6, 1, 8]
unknown = list(set(list(range(1, 9))) - set(known))
#%%
train_loader, test_loader, outloader = load_osr_data(options['data_dir'], known=known, unknown=unknown,
                                                     batch_size=options['batch_size'])
options['classes_legends'] = train_loader.dataset.get_labels_name()
options['num_classes'] = len(known)
#%%
feat_dim = 2
net = ARPLNet(feat_dim, options['num_classes'])
options['feat_dim'] = feat_dim
options['loss']="ARPLoss"
criterion = ARPLoss(**options)

if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    criterion = criterion.cuda()

if os.path.exists(options["save_dir"] + "/parm/net-" + options['name'] + ".parm"):
    print("Load exist parameter")
    net.load_state_dict(torch.load(options["save_dir"] + "/parm/net-" + options['name'] + ".parm"))
    criterion.load_state_dict(torch.load(options["save_dir"] + "/parm/crit-" + options['name'] + ".parm"))
else:
    print("Init Parameter RANDOM")
    init_weights(net)

params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
if options['eval']:
    print("==> Test", options['loss'])
    results = test(net, criterion, test_loader, outloader, epoch=0, use_gpu=options["use_gpu"])
    print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                            results['OSCR']))
    exit(0)
# %% Train
start_time = time.time()
for epoch in range(options['max_epoch']):
    print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
    T.train(net, criterion, optimizer, trainloader=train_loader, epoch=epoch, **options)

    if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
        torch.save(net.state_dict(), options["save_dir"] + "/parm/net-" + options['name'] + ".parm")
        torch.save(criterion.state_dict(), options["save_dir"] + "/parm/crit-" + options['name'] + ".parm")
        print("==> Test", options['loss'])
        results = test(net, criterion, test_loader, outloader, epoch=epoch,use_gpu=options["use_gpu"])
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                results['OSCR']))

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
