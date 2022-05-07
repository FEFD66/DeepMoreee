import argparse

import torch


def getCmdArgs(fakearg=False) :
    if fakearg:
        return getDefaultArgs()
    parser = argparse.ArgumentParser()

    # File path
    parser.add_argument('--save-dir', type=str, default='E:/more')
    parser.add_argument('--data-dir', type=str, default='E:')
    parser.add_argument('--log-dir', type=str, default='E:/log')

    # train
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument("--mode")
    parser.add_argument("--port")
    parser.add_argument("--eval", default=False, action='store_true')
    # parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    options: dict = vars(args)
    use_gpu = torch.cuda.is_available()
    device = "cuda:0"
    if options['cpu']:
        use_gpu = False
        device = "cpu"

    if use_gpu:
        print("Using GPU")
    else:
        print("Using CPU")
    options.update({
        'use_gpu': use_gpu,
        'device': device
    })
    return options


def getDefaultArgs():
    return {}
