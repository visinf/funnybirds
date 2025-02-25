import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import torchvision

from datasets.funny_birds import FunnyBirds

from models.resnet import resnet50
from models.vgg import vgg16
from models.bagnets.pytorchnet import bagnet33
from models.xdnns.xfixup_resnet import xfixup_resnet50

from models.bcos.model import get_model
from models.bcos.experiment_parameters import exps
from models.bcos.bcosconv2d import BcosConv2d
from models.ViT.ViT_new import vit_base_patch16_224

def setup_exp(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # create model
    if args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model.fc = torch.nn.Linear(2048, 50)

    elif args.model == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
        model.classifier[-1] = torch.nn.Linear(4096, 50)
    elif args.model == 'bagnet33':
        model = bagnet33(pretrained=args.pretrained)
        model.fc = torch.nn.Linear(2048, 50)
    elif args.model == 'x_resnet50':
        model = xfixup_resnet50()
        if args.pretrained:
            state_dict = torch.load(args.pretrained_ckpt)['state_dict']
            state_dict_new = {}
            for key in state_dict:
                new_key = key.replace('module.', "")
                state_dict_new[new_key] = state_dict[key]
            model.load_state_dict(state_dict_new)
            print('Model loaded')
        model.fc = torch.nn.Linear(2048, 50, bias=False)
    elif args.model == 'vit_b_16':
        model = vit_base_patch16_224(pretrained=args.pretrained)
        model.head = torch.nn.Linear(768, 50)
    elif args.model == 'bcos_resnet50':
        exp_params = exps["resnet_50"]
        model = get_model(exp_params)
        if args.pretrained:

            state_dict = torch.load(args.pretrained_ckpt)
            state_dict_new = {}
            for key in state_dict:
                new_key = key.replace('module.', "")
                state_dict_new[new_key] = state_dict[key]

            model.load_state_dict(state_dict_new)

            print('Model loaded')
        model[0].fc = BcosConv2d(2048, 50)
    else:
        print('Model not implemented')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model = model.to(device)

    # Data loading code -- add image size to 224 to make it 16x16
    need_resize = (args.model == 'vit_b_16')
    if need_resize:
        transforms = torch.nn.Sequential(
            torchvision.transforms.Resize((224, 224))
        )    
    else:
        transforms = None

    train_dataset = FunnyBirds(args.data, 'train', transform=transforms)
    test_dataset = FunnyBirds(args.data, 'test', transform=transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # define loss function (criterion), optimizer, and learning rate scheduler
    bcos_used = 'bcos' in args.model 
    if not bcos_used:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
        

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    return train_loader, test_loader, scheduler, optimizer, model, criterion