import argparse
import datetime
import time

import torch.nn
from more.datasets import load_more_data, load_data_mnist
from more.network import MoreNet, init_weights
import more.train as T
from more.ARPLoss import ARPLoss

parser = argparse.ArgumentParser()

# File path
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--data-dir', type=str, default='E:')

# train
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
# parser.add_argument('--gpu', type=str, default=True)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

options: dict = vars(args)

use_gpu = torch.cuda.is_available()
if options['cpu']:
    use = False

if use_gpu:
    print("Using GPU")
else:
    print("Using CPU")
options.update({
    'use_gpu': use_gpu
})

train_loader, test_loader = load_more_data(args['data_dir'], args['batch_size'])
args['classes_legends'] = train_loader.dataset.get_labels_name()
args['num_classes'] = len(args['classes_legends'])

net = MoreNet()
criterion = ARPLoss(**options)
if use_gpu:
    net=torch.nn.DataParallel(net).cuda()
    criterion=criterion.cuda()

params_list=[{'params':net.parameters()},{'params':criterion.parameters()}]
optimizer=torch.optim.SGD(params_list,lr=options['lr'],momentum=0.9,weight_decay=1e-4)


#%% Train
start_time= time.time()
for epoch in range(options['max_epoch']):
    print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
    T.train(net,criterion,optimizer,trainloader=train_loader,epoch=epoch,**options)

    if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
        print("Test(应该是，但还没实现)")
        # print("==> Test", options['loss'])
        # results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
        # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
        # save_networks(net, model_path, file_name, criterion=criterion)

elapsed = round(time.time() - start_time)
elapsed=str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
