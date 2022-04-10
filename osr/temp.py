import datetime
import time

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from core import train, test
from datasets.osr_dataloader import MNIST_OSR
from models import gan
from models.models import classifier32
from utils import save_networks, load_networks

#%%
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device="gpu:0"
data_root = "../data/mnist"
#%%
known = [2, 4, 5, 9, 8, 3]
unknown = list(set(list(range(0, 10))) - set(known))
batch_size=128
img_size= 32
Data = MNIST_OSR(known,data_root)
trainloader,testloader,outloader = Data.train_loader,Data.test_loader,Data.out_loader

#%%
net = classifier32(num_classes=Data.num_classes)
net=nn.DataParallel(net).cuda()
#%%
# GAN
nz,ns=100,1
netG = gan.Generator32(1, nz, 64, 3)
netD = gan.Discriminator32(1, 3, 64)
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

netG = nn.DataParallel(netG).cuda()
netD = nn.DataParallel(netD).cuda()
fixed_noise.cuda()
#%%
import loss.ARPLoss as Loss
criterion = Loss.ARPLoss(num_classes=Data.num_classes)
criterion=criterion.cuda()

criterionD = nn.BCELoss()
#%%

#%%
eval = False
model_path="E:\osr"
file_name= "parm"
#%%
if eval:
    net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
    results = test(net, criterion, testloader, outloader, epoch=0)
    print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t"
      .format(results['ACC'], results['AUROC'], results['OSCR']))
#%%
lr=0.1
gan_lr=0.0002
params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
optimizer = torch.optim.SGD(params_list, lr, momentum=0.9, weight_decay=1e-4)
# GAN
optimizerD = torch.optim.Adam(netD.parameters(), lr=gan_lr, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=gan_lr, betas=(0.5, 0.999))

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])
#%%
start_time = time.time()
max_epoch=20
eval_freq=1
print_freq=100
for epoch in range(max_epoch):
    print("==> Epoch {}/{}".format(epoch+1, max_epoch))

    # if options['cs']:
    # # train_cs(net, netD, netG, criterion, criterionD,
    #             optimizer, optimizerD, optimizerG,
    #             trainloader,nz,ns,print_freq,beta=0.1, epoch=epoch)
    #     train_cs(net, netD, netG, criterion, criterionD,
    #         optimizer, optimizerD, optimizerG,
    #         trainloader, epoch=epoch, **options)

    train(net, criterion, optimizer, trainloader, epoch=epoch,print_freq=print_freq)

    if eval_freq > 0 and (epoch+1) % eval_freq == 0 or (epoch+1) == max_epoch:
        print("==> Test", "ARPLoss")
        results = test(net, criterion, testloader, outloader, epoch=epoch)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        save_networks(net, model_path, file_name, criterion=criterion)

    if 30 > 0: scheduler.step()

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
#%%
