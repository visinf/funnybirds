import argparse
import torch
import torch.nn as nn
import random
import cv2

from torch.utils.data import DataLoader
from datasets.funny_birds import FunnyBirds
from models.vgg import vgg16

class CounterfactualVisualExplainer:
    
    def __init__(self, model, max_edits = 10):
        self.model = model
        self.max_edits = max_edits

    # features of shape 1, C, H, W
    def explain(self, features_query, features_distractor, class_query, class_distractor):
        B, C, H, W = features_query.shape
        assert B == 1
        
        #flatten features
        features_query_flat = features_query.view(C, H*W)
        features_distractor_flat = features_distractor.view(C, H*W)
                
        features_query_flat_edited = features_query_flat.clone()

        edits_query = []
        edits_distractor = []

        # each iteration is one edit
        for i in range(self.max_edits):    
            best_swap_idxs = (-1,-1) # index of features to swap
            best_swap_value = 0 # value of most increase from already visited swaps

            features_query_edited_before = features_query_flat_edited.clone().view(1, C, H, W)
            features_query_edited_before = features_query_edited_before.flatten(1)
            outputs_query_edited_before = self.model.classifier(features_query_edited_before)
            query_class_score_before_edit = outputs_query_edited_before[0,class_query].item()
            distractor_class_score_before_edit = outputs_query_edited_before[0,class_distractor].item()

            for f_query in range(features_query_flat.shape[-1]):
                if f_query in edits_query:
                    continue
                for f_distractor in range(features_distractor_flat.shape[-1]):
                    if f_distractor in edits_distractor:
                        continue
                        
                    features_query_flat_edited_tmp = features_query_flat_edited.clone()
                    features_query_flat_edited_tmp[:,f_query] = features_distractor_flat[:,f_distractor]

                    features_query_flat_edited_tmp = features_query_flat_edited_tmp.view(1, C, H, W)
                    features_query_flat_edited_tmp = features_query_flat_edited_tmp.flatten(1)
                    outputs_query_edited_tmp = self.model.classifier(features_query_flat_edited_tmp)
                    distractor_class_score_after_edit = outputs_query_edited_tmp[0,class_distractor].item()
                    
                    swap_value = distractor_class_score_after_edit - distractor_class_score_before_edit
                    if swap_value > best_swap_value:
                        best_swap_value = swap_value
                        best_swap_idxs = (f_query, f_distractor)

            edits_query.append(best_swap_idxs[0])
            edits_distractor.append(best_swap_idxs[1])
            features_query_flat_edited[:,best_swap_idxs[0]] = features_distractor_flat[:,best_swap_idxs[1]]
            features_query_flat_edited_after = features_query_flat_edited.clone().view(1, C, H, W)
            features_query_flat_edited_after = features_query_flat_edited_after.flatten(1)
            outputs_query_edited_after = self.model.classifier(features_query_flat_edited_after)
            class_edited = torch.argmax(outputs_query_edited_after).item()
            if class_edited == class_distractor: #class label already changed
                break
                
        return edits_query, edits_distractor, class_edited
    
    def get_edit_centers(self, edits, stride = 32):
        edit_centers = []
        for i in range(len(edits)):
            y = (edits[i] // 7) * stride + stride//2
            x = (edits[i] % 7) * stride + stride//2
            edit_centers.append((x,y))
        return edit_centers
    
def draw_bounding_boxes(image, centers, colors, stride = 32):
    for i, center in enumerate(centers):
        image = cv2.rectangle(image, (center[0]-stride//2,center[1]-stride//2), (center[0]+stride//2,center[1]+stride//2), colors(i), 1)
    return image



parser = argparse.ArgumentParser(description='FunnyBirds - Attribution Evaluation')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset (default: funnybird)')
parser.add_argument('--model', required=True,
                    choices=['vgg16'],
                    help='model architecture')
parser.add_argument('--checkpoint_name', type=str, required=True, default=None,
                    help='checkpoint name (including dir)')

parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--nr_itrs', default=250, type=int,
                    help='number of iterations')

parser.add_argument('--same_ori', default=False, action='store_true',
                    help='query and distractor images have same orientation')
parser.add_argument('--same_ori_one_diff', default=False, action='store_true',
                    help='query and distractor images have same orientation and only differ in one part')
parser.add_argument('--different_ori_one_diff', default=False, action='store_true',
                    help='query and distractor images have different orientation and only differ in one part')

def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = 'cuda:' + str(args.gpu)

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True) #BS of 2, one is query one is distractor

    # create model
    if args.model == 'vgg16':
        model = vgg16(num_classes=50)
    else:
        print('Model not implemented')
    

    model.load_state_dict(torch.load(args.checkpoint_name, map_location=torch.device('cpu'))['state_dict'])
    model.classifier = nn.Sequential(*list(model.classifier) + [nn.Softmax(1)])
    model = model.to(device)
    model.eval()

    colors_to_part = test_dataset.colors_to_part
    stride = 32


    cve = CounterfactualVisualExplainer(model, 49)

    nr_swaps_required = []
    nr_swaps_reasonable = [] # a swap is reasonable if it touches the same part in both images
    nr_different_parts = []
    itrs = 0
    while itrs < args.nr_itrs:
        for sample in test_dataloader:
            print(itrs)

            images = sample['image'].to(device)
            part_maps = sample['part_map'].to(device)
            params = sample['params']
            targets = sample['class_idx'].to(device)
            
            image_query = images[0].unsqueeze(0)[:,:,16:-16,16:-16]
            image_distractor = images[1].unsqueeze(0)[:,:,16:-16,16:-16]
            
            part_map_query = part_maps[0].unsqueeze(0)[:,:,16:-16,16:-16]
            part_map_distractor = part_maps[1].unsqueeze(0)[:,:,16:-16,16:-16]
            
            class_query = targets[0].item()
            class_distractor = targets[1].item()
            
            if class_query == class_distractor:
                continue

            if args.same_ori:
                params_query = test_dataset.get_params_for_single(params, 0)
                params_distractor = test_dataset.get_params_for_single(params, 1)
                params_distractor['camera_distance'] = params_query['camera_distance']
                params_distractor['camera_pitch'] = params_query['camera_pitch']
                params_distractor['camera_roll'] = params_query['camera_roll']
                params_distractor['light_distance'] = params_query['light_distance']
                params_distractor['light_pitch'] = params_query['light_pitch']
                params_distractor['light_roll'] = params_query['light_roll']
                image_distractor, _ = test_dataset.render(params_distractor)
                image_distractor = image_distractor.to(device)
                image_distractor = image_distractor[:,:,16:-16,16:-16]

            if args.same_ori_one_diff:
                close_classes = test_dataset.get_classes_with_distance_n(class_query, 1)
                if len(close_classes) == 0:
                    continue
                class_distractor = random.choice(close_classes)

                params_query = test_dataset.get_params_for_single(params, 0)
                params_distractor = test_dataset.get_params_for_single(params, 1)
                params_distractor['camera_distance'] = params_query['camera_distance']
                params_distractor['camera_pitch'] = params_query['camera_pitch']
                params_distractor['camera_roll'] = params_query['camera_roll']
                params_distractor['light_distance'] = params_query['light_distance']
                params_distractor['light_pitch'] = params_query['light_pitch']
                params_distractor['light_roll'] = params_query['light_roll']
                image_distractor, _ = test_dataset.render_class(class_distractor, params_distractor)
                image_distractor = image_distractor.to(device)
                image_distractor = image_distractor[:,:,16:-16,16:-16]

            if args.different_ori_one_diff:
                close_classes = test_dataset.get_classes_with_distance_n(class_query, 1)
                if len(close_classes) == 0:
                    continue
                class_distractor = random.choice(close_classes)
                params_distractor = test_dataset.get_params_for_single(params, 1)
                image_distractor, _ = test_dataset.render_class(class_distractor, params_distractor)
                image_distractor = image_distractor.to(device)
                image_distractor = image_distractor[:,:,16:-16,16:-16]

            pred_query = model(image_query).argmax(dim=1).item()
            pred_distractor = model(image_distractor).argmax(dim=1).item()
            if pred_query != class_query or pred_distractor != class_distractor:
                continue


            features_query = model.features(image_query)
            features_distractor = model.features(image_distractor)

            swaps_i1, swaps_i2, class_edited = cve.explain(features_query, features_distractor, class_query, class_distractor)

            if len(swaps_i1) == 49:
                continue

            nr_swaps_required.append(len(swaps_i1))
            nr_different_parts.append(test_dataset.class_distance(class_query, class_distractor))

            edit_centers_query = cve.get_edit_centers(swaps_i1)
            edit_centers_distractor = cve.get_edit_centers(swaps_i2) 

            parts_im_rects_query = []
            for edit_center in edit_centers_query:
                parts_in_rect = []
                part_rectangle = part_map_query[0,:,edit_center[1]-stride//2:edit_center[1]+stride//2, edit_center[0]-stride//2:edit_center[0]+stride//2]
                part_rectangle = part_rectangle
                for part_color in colors_to_part.keys():
                    torch_color = torch.zeros(3,1,1).to(device)
                    torch_color[0,0,0] = part_color[0]
                    torch_color[1,0,0] = part_color[1]
                    torch_color[2,0,0] = part_color[2]
                    color_available = torch.any(torch.all(part_rectangle == torch_color, dim = 0))
                    if color_available:
                        part = colors_to_part[part_color]
                        part = ''.join([i for i in part if not i.isdigit()]) #remove numbers because we don't distinguish between left and right part
                        parts_in_rect.append(part)
                parts_im_rects_query.append(list(set(parts_in_rect)))
            
            parts_im_rects_distractor = []
            for edit_center in edit_centers_distractor:
                parts_in_rect = []
                part_rectangle = part_map_distractor[0,:,edit_center[1]-stride//2:edit_center[1]+stride//2, edit_center[0]-stride//2:edit_center[0]+stride//2]
                part_rectangle = part_rectangle
                for part_color in colors_to_part.keys():
                    torch_color = torch.zeros(3,1,1).to(device)
                    torch_color[0,0,0] = part_color[0]
                    torch_color[1,0,0] = part_color[1]
                    torch_color[2,0,0] = part_color[2]
                    color_available = torch.any(torch.all(part_rectangle == torch_color, dim = 0))
                    if color_available:
                        part = colors_to_part[part_color]
                        part = ''.join([i for i in part if not i.isdigit()]) #remove numbers because we don't distinguish between left and right part
                        parts_in_rect.append(part)
                parts_im_rects_distractor.append(list(set(parts_in_rect)))
            
            nr_reasonable = 0
            for i in range(len(swaps_i1)):
                parts_im_rect_distractor = parts_im_rects_distractor[i]
                parts_im_rect_query = parts_im_rects_query[i]
                for part in parts_im_rect_query:
                    if part in parts_im_rect_distractor:
                        nr_reasonable += 1
                        break
                
            nr_swaps_reasonable.append(nr_reasonable)

            itrs += 1

            if itrs == args.nr_itrs:
                break
            
    print(nr_swaps_required)
    print(nr_swaps_reasonable)
    print(nr_different_parts)

    print(sum(nr_swaps_required) / len(nr_swaps_required))
    print(sum(nr_swaps_reasonable) / len(nr_swaps_reasonable))
    print('Unreasonable swaps:')
    print(sum(nr_swaps_required) / len(nr_swaps_required) - sum(nr_swaps_reasonable) / len(nr_swaps_reasonable))
    
    print(sum(nr_different_parts) / len(nr_different_parts))

if __name__ == '__main__':
    main()