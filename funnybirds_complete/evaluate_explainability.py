# python evaluate_explainability.py --data /path/to/datasets/FunnyBirds --model resnet50 --explainer IntegratedGradients --checkpoint_name /path/to/FunnyBirds/checkpoints/resnet50_default_checkpoint_best.pth.tar --nr_itrs 3 --distractibility
# python evaluate_explainability.py --data /path/to/datasets/FunnyBirds --model resnet50 --explainer InputXGradient --checkpoint_name /path/to/FunnyBirds/checkpoints/resnet50_default_checkpoint_best.pth.tar --accuracy
# python evaluate_explainability.py --data /path/to/datasets/FunnyBirds --model resnet50 --explainer Saliency --checkpoint_name /path/to/FunnyBirds/checkpoints/resnet50_default_checkpoint_best.pth.tar --controlled_synthetic_data_check

import argparse
import random
import torch
from captum.attr import IntegratedGradients, LayerGradCam, InputXGradient, Saliency, GuidedGradCam, GuidedBackprop, DeepLift

from models.resnet import resnet50
from models.vgg import vgg16
from models.ViT.ViT_new import vit_base_patch16_224
from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from models.protopnet.ppnet import ppnetexplain

from models.model_wrapper import StandardModel, ViTModel, BcosModel, ProtoPNetModel
from evaluation_protocols import accuracy_protocol, controlled_synthetic_data_check_protocol, single_deletion_protocol, preservation_check_protocol, deletion_check_protocol, target_sensitivity_protocol, distractibility_protocol, background_independence_protocol
from explainers.explainer_wrapper import CaptumAttributionExplainer, BcosExplainer, ProtoPNetExplainer, RiseExplainer, LimeExplainer, BagNetExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, IntegratedGradientsAbsoluteExplainer, SSMExplainer, SSMAttriblikePExplainer, ViTGuidedGradCamExplainer
from models.bcos.model import get_model
from models.bcos.experiment_parameters import exps
from models.bcos.bcosconv2d import BcosConv2d

from models.bagnets.pytorchnet import bagnet33
from models.xdnns.xfixup_resnet import xfixup_resnet50

import models.protopnet.model as model_ppnet


parser = argparse.ArgumentParser(description='FunnyBirds - Explanation Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--model', required=True,
                    choices=['resnet50', 'vgg16', 'bcos_resnet50', 'bagnet33', 'x_resnet50', 'protopnet_resnet50', 'vit_b_16'],
                    help='model architecture')
parser.add_argument('--explainer', required=True,
                    choices=['IntegratedGradients', 'InputXGradient', 'Saliency', 'Rise', 'Lime', 'Bcos', 'BagNet', 'GradCam', 'ProtoPNet', 'Rollout', 'CheferLRP', 'IntegratedGradientsAbsolute', 'SSMExplainer', 'SSMAttriblikePExplainer', 'GuidedGradCam', 'GuidedBackprop', 'DeepLift'],
                    help='explainer')
parser.add_argument('--checkpoint_name', type=str, required=True, default=None,
                    help='checkpoint name (including dir)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='seed')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
parser.add_argument('--nr_itrs', default=2501, type=int,
                    help='batch size for protocols that do not require custom BS such as accuracy')
                    
parser.add_argument('--accuracy', default=False, action='store_true',
                    help='compute accuracy')

parser.add_argument('--controlled_synthetic_data_check', default=False, action='store_true',
                    help='compute controlled synthetic data check')

parser.add_argument('--single_deletion', default=False, action='store_true',
                    help='compute single deletion')

parser.add_argument('--preservation_check', default=False, action='store_true',
                    help='compute preservation check')

parser.add_argument('--deletion_check', default=False, action='store_true',
                    help='compute deletion check')

parser.add_argument('--target_sensitivity', default=False, action='store_true',
                    help='compute target sensitivity')

parser.add_argument('--distractibility', default=False, action='store_true',
                    help='compute distractibility')

parser.add_argument('--background_independence', default=False, action='store_true',
                    help='compute background dependence')

def main():
    args = parser.parse_args()
    device = 'cuda:' + str(args.gpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    if args.model == 'resnet50':
        model = resnet50(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vgg16':
        model = vgg16(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'vit_b_16':
        if args.explainer == 'CheferLRP':
            model = vit_LRP(num_classes=50)
        else:
            model = vit_base_patch16_224(num_classes = 50)
        model = ViTModel(model)
    elif args.model == 'bcos_resnet50':
        exp_params = exps["resnet_50"]
        model = get_model(exp_params)
        model[0].fc = BcosConv2d(512 * 4, 50)
        model = BcosModel(model)
    elif args.model == 'bagnet33':
        model = bagnet33(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'x_resnet50':
        model = xfixup_resnet50(num_classes = 50)
        model = StandardModel(model)
    elif args.model == 'protopnet_resnet50':
        base_architecture = 'resnet50'
        img_size = 256
        prototype_shape = (50*10, 128, 1, 1)
        num_classes = 50
        prototype_activation_function = 'log' 
        add_on_layers_type = 'regular'
        load_model_dir = '/path/to/model_weights/FunnyBirds/protopnet/saved_models/resnet50/007'
        epoch_number_str = '60'
        print('REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH')
        model = model_ppnet.construct_PPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
        model = ProtoPNetModel(model, load_model_dir, epoch_number_str)
    else:
        print('Model not implemented')
    
    if args.model != 'protopnet_resnet50':
        model.load_state_dict(torch.load(args.checkpoint_name, map_location=torch.device('cpu'))['state_dict'])
    else:
        state_dict = torch.load(args.checkpoint_name, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # create explainer
    if args.explainer == 'InputXGradient':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'Saliency':
        explainer = Saliency(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'IntegratedGradients':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,256,256)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline)
    elif args.explainer == 'IntegratedGradientsAbsolute':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,256,256)).to(device)
        explainer = IntegratedGradientsAbsoluteExplainer(explainer, baseline=baseline)
    elif args.explainer == 'Rise':
        explainer = RiseExplainer(model)
    elif args.explainer == 'Lime':
        explainer = LimeExplainer(model)
    elif args.explainer == 'Bcos':
        explainer = BcosExplainer(model)
    elif args.explainer == 'BagNet':
        explainer = BagNetExplainer(model)
    elif args.explainer == 'GradCam':
        if args.model == 'resnet50':
            explainer = LayerGradCam(model, model.model.layer4)
            explainer = CaptumAttributionExplainer(explainer)
        elif args.model == 'vgg16':
            explainer = LayerGradCam(model, model.model.features)
            explainer = CaptumAttributionExplainer(explainer)
        elif args.model == 'vit_b_16':
            explainer = ViTGradCamExplainer(model)
        else:
            print('GradCAM not supported for model!')
            return
    elif args.explainer == 'ProtoPNet':
        explainer = ProtoPNetExplainer(model)
    elif args.explainer == 'Rollout':
        explainer = ViTRolloutExplainer(model)
    elif args.explainer == 'CheferLRP':
        explainer = ViTCheferLRPExplainer(model)
    elif args.explainer == 'SSMExplainer':
        explainer = ppnetexplain(model)
        explainer = SSMExplainer(explainer)
    elif args.explainer == 'SSMAttriblikePExplainer':
        explainer = ppnetexplain(model)
        explainer = SSMAttriblikePExplainer(explainer)
    elif args.explainer == 'GuidedGradCam':
        if args.model == 'resnet50':
            explainer = GuidedGradCam(model, model.model.layer4)
            explainer = CaptumAttributionExplainer(explainer)
        elif args.model == 'vgg16':
            explainer = GuidedGradCam(model, model.model.features)
            explainer = CaptumAttributionExplainer(explainer)
        elif args.model == 'vit_b_16':
            explainer = ViTGuidedGradCamExplainer(model)
        else:
            print('GuidedGradCAM not supported for model!')
            return

    elif args.explainer == 'GuidedBackprop':
        explainer = GuidedBackprop(model)
        explainer = CaptumAttributionExplainer(explainer)
    elif args.explainer == 'DeepLift':
        explainer = DeepLift(model)
        explainer = CaptumAttributionExplainer(explainer)
    else:
        print('Explainer not implemented')

    accuracy, csdc, pc, dc, distractibility, sd, ts, background_independence = -1, -1, -1, -1, -1, -1, -1, -1

    if args.accuracy:
        print('Computing accuracy...')
        accuracy = accuracy_protocol(model, args)
        accuracy = round(accuracy, 5)

    if args.controlled_synthetic_data_check:
        print('Computing controlled synthetic data check...')
        csdc = controlled_synthetic_data_check_protocol(model, explainer, args)

    if args.target_sensitivity:
        print('Computing target sensitivity...')
        ts = target_sensitivity_protocol(model, explainer, args)
        ts = round(ts, 5)

    if args.single_deletion:
        print('Computing single deletion...')
        sd = single_deletion_protocol(model, explainer, args)
        sd = round(sd, 5)

    if args.preservation_check:
        print('Computing preservation check...')
        pc = preservation_check_protocol(model, explainer, args)

    if args.deletion_check:
        print('Computing deletion check...')
        dc = deletion_check_protocol(model, explainer, args)

    if args.distractibility:
        print('Computing distractibility...')
        distractibility = distractibility_protocol(model, explainer, args)

    if args.background_independence:
        print('Computing background independence...')
        background_independence = background_independence_protocol(model, args)
        background_independence = round(background_independence, 5)
    
    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = csdc[threshold]/3. + pc[threshold]/3. + dc[threshold]/3. + distractibility[threshold]
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print('FINAL RESULTS:')
    keys = ['Accuracy', 'CSDC', 'PC', 'DC', 'Distractability', 'Background independence', 'SD', 'TS']
    results = [accuracy, csdc, pc, dc, distractibility, background_independence, sd, ts]
    for k in range(len(keys)):
        res = results[k]
        if res != -1:
            if not isinstance(res, dict):
                print(keys[k], ': ', res)
            else:
                print(keys[k], ': ', round(res[best_threshold],5))

    print('Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS')
    print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(accuracy, round(csdc[best_threshold],5), round(pc[best_threshold],5), round(dc[best_threshold],5), round(distractibility[best_threshold],5), background_independence, sd, ts))
    print('Best threshold:', best_threshold)

if __name__ == '__main__':
    main()
