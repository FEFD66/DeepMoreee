import os

import torch

from more.network import init_weights


def init_net(net, options):
    init_parm(net, "net", options)


def init_criterion(criterion, options):
    init_parm(criterion, "crit", options)


def save_net(net, options):
    save(net, "net", options)


def save_criterion(net, options):
    save(net, "crit", options)


def save(net, prefix: str, options:dict):
    file = __get_file(options["save_dir"], prefix, options['name'])
    print("Saving " + prefix + " at " + file)
    torch.save(net.state_dict(), file)


def init_parm(net, prefix: str, options: dict):
    file = __get_file(options["save_dir"], prefix, options['name'])
    if os.path.exists(file):
        print("Load exist parameter for " + prefix)
        net.load_state_dict(torch.load(file))
    else:
        print("Init " + prefix + " Parameter RANDOM")
        init_weights(net)


def __get_file(path: str, prefix: str, name: str):
    path = path + "/parm"
    __mkdir(path)
    return path + "/" + prefix + "-" + name + ".parm"


def __mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
