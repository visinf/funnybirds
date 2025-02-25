import os


#import sys
#sys.path.append('~/code/phd_interventional_xai/python/B-cos')

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel, DataParallel

#from ....models.baselines.pretrained import MyVGG11, MyResNet34, MyDenseNet121, MyInception
#from ....models.bcos.densenet import densenet121
#from ....models.bcos.inception import inception_v3
from torch.hub import download_url_to_file
from .resnet import resnet34, resnet50
#from ....models.bcos.vgg import vgg11
from .utils import FinalLayer, MyAdaptiveAvgPool2d
from importlib import import_module

def get_exp_params(save_path):
    """
    Retrieve the experiments specs by parsing the path. The experiments all follow the same naming
    and save_path convention, so that reloading a trainer from a path is easy.
    The experiments are imported with importlib.
    Args:
        save_path: Path to the experiment results folder. Should be of the form
            '...experiments/dataset/base_net/experiment_name'.
    Returns: Experiment specifications as dict.
    """
    base_dir, dataset, base_net, exp_name = save_path.split("/")[-4:]
    exp_params_module = ".".join([base_dir, dataset, base_net, "experiment_parameters"])
    exp_params_module = import_module(exp_params_module)
    exp_params = exp_params_module.get_exp_params(exp_name)
    return exp_params


archs = {
    #"densenet_121": densenet121,
    #"inception_v3": inception_v3,
    #"vgg_11": vgg11,
    "resnet_34": resnet34,
    "resnet_50": resnet50,
             }


def get_pretrained_model(exp_params):

    model = {
        "vgg11": MyVGG11,
        "resnet34": MyResNet34,
        "inception": MyInception,
        "densenet121": MyDenseNet121,
    }[exp_params["network"]]()
    network_list = [model]
    network_list += [
        MyAdaptiveAvgPool2d((1, 1)),
        FinalLayer(bias=0, norm=1)
    ]
    network = nn.Sequential(*network_list)
    network.opti = exp_params["opti"]
    network.opti_opts = exp_params["opti_opts"]

    return network


def load_pretrained(exp_params, network):
    model_path = os.path.join("bcos_pretrained", exp_params["exp_name"])
    model_file = os.path.join(model_path, "state_dict.pkl")

    if not os.path.exists(model_file):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        download_url_to_file(exp_params["model_url"], model_file)
    loaded_state_dict = torch.load(model_file, map_location="cpu")
    network.load_state_dict(loaded_state_dict)


def get_model(exp_params):
    #if exp_params.get("pretrained", False):
    #    return get_pretrained_model(exp_params)
    logit_bias = exp_params["logit_bias"]
    logit_temperature = exp_params["logit_temperature"]
    network = archs[exp_params["network"]]
    network_opts = exp_params["network_opts"]
    network_list = [network(**network_opts)]

    network_list += [
        MyAdaptiveAvgPool2d((1, 1)),
        FinalLayer(bias=logit_bias, norm=logit_temperature)
    ]
    network = nn.Sequential(*network_list)
    #if exp_params["load_pretrained"]:
    #    load_pretrained(exp_params, network)
    network.opti = exp_params["opti"]
    network.opti_opts = exp_params["opti_opts"]
    return network


def get_optimizer(model, base_lr):
    optimisers = {"Adam": torch.optim.Adam,
                  "AdamW": torch.optim.AdamW,
                  "SGD": torch.optim.SGD}
    the_model = model if not isinstance(model, DistributedDataParallel) and not isinstance(model, DataParallel) else model.module
        
    opt = optimisers[the_model.opti]
    opti_opts = the_model.opti_opts
    opti_opts.update({"lr": base_lr})
    opt = opt(the_model.parameters(), **opti_opts)
    opt.base_lr = base_lr
    return opt