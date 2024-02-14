import argparse
import os
import torch
import numpy as np
import random
import cv2

from torch.utils.data import DataLoader

from datasets.funny_birds import FunnyBirds
import models.protopnet.model as model_ppnet
from models.model_wrapper import ProtoPNetModel
from explainers.explainer_wrapper import ProtoPNetExplainer


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


parser = argparse.ArgumentParser(description='FunnyBirds - Attribution Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--load_model_dir', metavar='DIR', required=True,
                    help='path to dataset (default: imagenet)')
parser.add_argument('--epoch_number_str', type=str, required=True,
                    help='GPU id to use.'),
parser.add_argument('--checkpoint_name', type=str, required=True, default=None,
                    help='checkpoint name (including dir)'),
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--nr_itrs', default=250, type=int,
                    help='number of iterations')


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda:' + str(args.gpu)

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    train_dataset = FunnyBirds(args.data, 'train', get_part_map=True)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    base_architecture = 'resnet50'
    img_size = 256
    prototype_shape = (50*10, 128, 1, 1)
    num_classes = 50
    prototype_activation_function = 'log' 
    add_on_layers_type = 'regular'
    load_model_dir = args.load_model_dir
    epoch_number_str = args.epoch_number_str
    load_img_dir = os.path.join(load_model_dir, 'img')
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))

    print('REMEMBER TO ADJUST PROTOPNET PATH AND EPOCH')
    model = model_ppnet.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    model = ProtoPNetModel(model, load_model_dir, epoch_number_str)
    
    state_dict = torch.load(args.checkpoint_name, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    explainer = ProtoPNetExplainer(model)
    

    colors_to_part = test_dataset.colors_to_part

    nr_total_prototypes = 0
    nr_reasonable_prototypes = 0
    nr_empty_prototypes_test = 0
    nr_empty_prototypes_train = 0
    nr_parts_prototypes_test = {}
    nr_parts_prototypes_train = {}
    
    nr_parts_prototypes_test['0'] = 0
    nr_parts_prototypes_test['1'] = 0
    nr_parts_prototypes_test['2'] = 0
    nr_parts_prototypes_test['3'] = 0
    nr_parts_prototypes_test['4'] = 0
    nr_parts_prototypes_test['5'] = 0

    nr_parts_prototypes_train['0'] = 0
    nr_parts_prototypes_train['1'] = 0
    nr_parts_prototypes_train['2'] = 0
    nr_parts_prototypes_train['3'] = 0
    nr_parts_prototypes_train['4'] = 0
    nr_parts_prototypes_train['5'] = 0
    
    
    itrs = 0
    while itrs < args.nr_itrs:
        for sample in test_dataloader:
            print(itrs)

            images = sample['image'].to(device)
            part_map_test = sample['part_map'].to(device)
            
            targets = sample['class_idx'].to(device)
            
            inference_image_masks, similarity_scores, class_connections, prototypes, bounding_box_coords, prototype_idxs = explainer.explain(images, target=targets)

            for i in range(0,10):
                
                prototype_idx = prototype_idxs[i]
                bbox_height_start=prototype_info[prototype_idx.item()][1]
                bbox_height_end=prototype_info[prototype_idx.item()][2]
                bbox_width_start=prototype_info[prototype_idx.item()][3]
                bbox_width_end=prototype_info[prototype_idx.item()][4]

                p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch_number_str), 'prototype-img-original'+str(prototype_idx.item())+'.png'))
                image_prototype = p_img_bgr[...,::-1] # rgb conversion
                image_prototype = torch.from_numpy(image_prototype.copy()).permute(2,0,1).unsqueeze(0)
                image_prototype = image_prototype.type(torch.LongTensor)
                
                match_found = False
                print('searching image match...')
                train_dataset_for_target = torch.utils.data.Subset(train_dataset, list(range(targets[0]*1000, (targets[0]+1)*1000-1)))
                train_loader = torch.utils.data.DataLoader(train_dataset_for_target, batch_size=1, shuffle=False)
                for sample_train in train_loader:
                    image_train = sample_train['image'].to(device)
                    target_train = sample_train['class_idx'].to(device)
                    #print(target_train)
                    #if target_train != targets:
                    #    continue
                    image_train = (image_train*255.).type(torch.LongTensor)
                    part_map_train = sample_train['part_map'].to(device)
                    
                    if (image_train == image_prototype).all():
                        print('MATCH FOUND')
                        match_found = True
                        break
                if not match_found:
                    print('No match was found :( ')
                    return
                    continue
                
                print('Parts in prototype (in train image):')
                part_rectangle = part_map_train[0,:,bbox_height_start:bbox_height_end-1, bbox_width_start:bbox_width_end-1]

                colors_to_part = test_dataset.colors_to_part

                parts_in_rect_train = []
                for part_color in colors_to_part.keys():
                    torch_color = torch.zeros(3,1,1).to(device)
                    torch_color[0,0,0] = part_color[0]
                    torch_color[1,0,0] = part_color[1]
                    torch_color[2,0,0] = part_color[2]
                    color_available = torch.any(torch.all(part_rectangle == torch_color, dim = 0))
                    if color_available:
                        part = colors_to_part[part_color]
                        part = ''.join([i for i in part if not i.isdigit()]) #remove numbers because we don't distinguish between left and right part
                        parts_in_rect_train.append(part)

                print(parts_in_rect_train)
                
                
                
                print('Parts in prototype (in test image):')
                # find parts in test image of prototype

                # show the most highly activated patch of the image by this prototype
                high_act_patch_indices = bounding_box_coords[i]
                
                part_rectangle = part_map_test[0,:,high_act_patch_indices[0]:high_act_patch_indices[1]-1, high_act_patch_indices[2]:high_act_patch_indices[3]-1]

                colors_to_part = test_dataset.colors_to_part

                parts_in_rect_test = []
                for part_color in colors_to_part.keys():
                    torch_color = torch.zeros(3,1,1).to(device)
                    torch_color[0,0,0] = part_color[0]
                    torch_color[1,0,0] = part_color[1]
                    torch_color[2,0,0] = part_color[2]
                    color_available = torch.any(torch.all(part_rectangle == torch_color, dim = 0))
                    if color_available:
                        part = colors_to_part[part_color]
                        part = ''.join([i for i in part if not i.isdigit()]) #remove numbers because we don't distinguish between left and right part
                        parts_in_rect_test.append(part)

                print(parts_in_rect_test)

                #parts_in_rect_train = list(dict.fromkeys(parts_in_rect_train))
                #parts_in_rect_test = list(dict.fromkeys(parts_in_rect_test))

                intersection = list(set(parts_in_rect_train) & set(parts_in_rect_test))
                print('Overlap', intersection)
                if len(intersection) > 0:
                    nr_reasonable_prototypes += 1
                nr_total_prototypes += 1
                if len(parts_in_rect_train) == 0:
                    nr_empty_prototypes_train += 1
                if len(parts_in_rect_test) == 0:
                    nr_empty_prototypes_test += 1

                nr_parts_prototypes_train[str(len(list(set(parts_in_rect_train))))] += 1
                nr_parts_prototypes_test[str(len(list(set(parts_in_rect_test))))] += 1

            itrs += 1

            if itrs == args.nr_itrs:
                break

    print('nr_total_prototypes', nr_total_prototypes)
    print('nr_reasonable_prototypes', nr_reasonable_prototypes)
    print('nr_empty_prototypes_train', nr_empty_prototypes_train)
    print('nr_empty_prototypes_test', nr_empty_prototypes_test)
    print(nr_parts_prototypes_train)
    print(nr_parts_prototypes_test)


if __name__ == '__main__':
    main()