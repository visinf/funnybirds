import torch
import torch.nn as nn
from abc import abstractmethod
import os
import numpy as np
import cv2

class AbstractExplainer():
    def __init__(self, explainer, baseline = None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        print(self.explainer_name)

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)

class AbstractAttributionExplainer(AbstractExplainer):
    
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)



    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg = False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1 # B = 1
        attribution = self.explain(image, target=target)
        #m = nn.ReLU()
        #positive_attribution = m(attribution)

        part_importances = self.get_part_importance(image, part_map, target, colors_to_part, with_bg = with_bg)
        #total_attribution_in_parts = 0
        #for key in part_importances.keys():
        #    total_attribution_in_parts += abs(part_importances[key])

        important_parts_for_thresholds = []
        for threshold in thresholds:
            important_parts = []
            for key in part_importances.keys():
                if part_importances[key] > (attribution.sum() * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds



    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg = False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1 # B = 1
        attribution = self.explain(image, target=target)
        
        part_importances = {}

        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        #dilation1 = nn.MaxPool2d(25, stride=1, padding=12)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1,3,1,1).to(image.device)
            torch_color[0,0,0,0] = part_color[0]
            torch_color[0,1,0,0] = part_color[1]
            torch_color[0,2,0,0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
            
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            #attribution_in_part = attribution_in_part.sum() / (color_available_dilated.sum() + 1e-8)
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50): # TODO: adjust 50 if more background parts are used
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = 204
                torch_color[0,1,0,0] = 204
                torch_color[0,2,0,0] = 204+i
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                #attribution_in_part = attribution_in_part.sum() / (color_available_dilated.sum() + 1e-8)
                attribution_in_part = attribution_in_part.sum()
                
                bg_string = 'bg_' + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

import matplotlib.pyplot as plt
# this should in the end be the final explainer
# this explainer can also be used for the visualizations to clean up the code a bit
class ProtoPNetExplainer(AbstractAttributionExplainer):
    """
    A wrapper for ProtoPNet.
    Args:
        model: PyTorch model.
    """
    def __init__(self, model):
        """
        A wrapper for ProtoPNet explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model
        self.load_model_dir = model.load_model_dir #'/data/rhesse/FunnyBirds/protopnet/saved_models/resnet50/004'
        #self.load_model_path = '20nopush0.9360.pth'
        self.load_img_dir = os.path.join(self.load_model_dir, 'img')
        #load_model_name = 'TODO' #'10_18push0.7822.pth'
        self.epoch_number_str = model.epoch_number_str # '90'
        self.start_epoch_number = int(self.epoch_number_str)

    def find_high_activation_crop(self, activation_map, percentile=95):
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

    # for evaluating protopnet explainations are masks
    def explain(self, image, target):
        aggr_patches_w_impact = torch.zeros_like(image)       
        B,C,H,W = image.shape
        image_numpy = image[0].permute(1,2,0).cpu().numpy()
        original_img = image_numpy
            
        prototype_info = np.load(os.path.join(self.load_img_dir, 'epoch-'+self.epoch_number_str, 'bb'+self.epoch_number_str+'.npy'))
        prototype_img_identity = prototype_info[:, -1]

        idx = 0

        logits, min_distances = self.model(image, return_min_distances = True)
        conv_output, distances = self.model.model.push_forward(image)
        prototype_activations = self.model.model.distance_2_similarity(min_distances)
        prototype_activation_patterns = self.model.model.distance_2_similarity(distances)
        
        array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
            
        topk_classes = target
        c = topk_classes[0]
        i = 0

        class_prototype_indices = np.nonzero(self.model.model.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1
        
        inference_image_masks = []
        prototypes = [] # these are the training set prototypes
        prototype_idxs = [] # these are the training set prototypes
        similarity_scores = []
        class_connections = []
        bounding_box_coords = []

        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            prototype_idxs.append(prototype_index)
            
            prototype = plt.imread(os.path.join(self.load_img_dir, 'epoch-'+str(self.start_epoch_number), 'prototype-img'+str(prototype_index.item())+'.png'))
            prototype = cv2.cvtColor(np.uint8(255*prototype), cv2.COLOR_RGB2BGR)
            
            h,w,c = prototype.shape
            prototype = prototype[...,::-1]
            
            prototypes.append(prototype)

            
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(H, W),
                                                      interpolation=cv2.INTER_CUBIC)
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = self.find_high_activation_crop(upsampled_activation_pattern)

            mask = torch.zeros_like(image)
            mask[:,:,high_act_patch_indices[0]:high_act_patch_indices[1]-1, high_act_patch_indices[2]:high_act_patch_indices[3]-1] = 1
            
            inference_image_masks.append(mask)
            similarity_scores.append(prototype_activations[idx][prototype_index]) 
            class_connections.append(self.model.model.last_layer.weight[c][prototype_index])
            bounding_box_coords.append(high_act_patch_indices)

            prototype_cnt += 1

    
        return inference_image_masks, similarity_scores, class_connections, prototypes, bounding_box_coords, prototype_idxs

    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg = False):
        assert image.shape[0] == 1 # B = 1
        # explain
        inference_image_masks, similarity_scores, class_connections, _, _, _ = self.explain(image, target)
        attribution = torch.zeros_like(image)
        for inference_image_mask, similarity_score, class_connection in zip(inference_image_masks, similarity_scores, class_connections):
            inference_image_mask = inference_image_mask.to(image.device)
            attribution = attribution + inference_image_mask * similarity_score * class_connection

        part_importances = {}

        dilation1 = nn.MaxPool2d(1, stride=1, padding=0)
        #dilation1 = nn.MaxPool2d(25, stride=1, padding=12)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1,3,1,1).to(image.device)
            torch_color[0,0,0,0] = part_color[0]
            torch_color[0,1,0,0] = part_color[1]
            torch_color[0,2,0,0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
            
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            #attribution_in_part = attribution_in_part.sum() / (color_available_dilated.sum() + 1e-8)
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50): # TODO: adjust 50 if more background parts are used
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = 204
                torch_color[0,1,0,0] = 204
                torch_color[0,2,0,0] = 204+i
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                #attribution_in_part = attribution_in_part.sum() / (color_available_dilated.sum() + 1e-8)
                attribution_in_part = attribution_in_part.sum()
                
                bg_string = 'bg_' + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances
    


    

    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg = False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1 # B = 1
        # explain
        inference_image_masks, similarity_scores, class_connections, _, _, _ = self.explain(image, target)
        attribution = torch.zeros_like(image)
        for inference_image_mask in inference_image_masks:
            inference_image_mask = inference_image_mask.to(image.device)
            attribution = attribution + inference_image_mask

        attribution = attribution.clamp(min = 0., max=1.)
        
        important_parts_for_thresholds = []
        
        
        for threshold in thresholds:
            important_parts = []
            for part_color in colors_to_part.keys():
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = part_color[0]
                torch_color[0,1,0,0] = part_color[1]
                torch_color[0,2,0,0] = part_color[2]
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                #color_available_dilated = dilation1(color_available)
                #surroundings = dilation2(color_available)
                #surroundings = surroundings - color_available_dilated # these are pixels around the part but without the part
                attribution_in_part = attribution * color_available
                attribution_in_part = attribution_in_part.sum()
                #attribution_in_part = attribution_in_part.sum() / color_available_dilated.sum()
                #attribution_around_part = attribution * surroundings
                #attribution_around_part = attribution_around_part.sum() / surroundings.sum()
                #if attribution_in_part > attribution_around_part * 1.5: #factor to decide how much bigger attribution in part should be
                #    important_parts.append(colors_to_part[part_color])
                
                if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                    important_parts.append(colors_to_part[part_color])

            important_parts = list(map(lambda part_string: ''.join((x for x in part_string if not x.isdigit())), important_parts)) # remove 01 and 02 from parts
            important_parts = list(dict.fromkeys(important_parts)) # remove duplicates, e.g. feet, feet
            important_parts_for_thresholds.append(important_parts)
        

        if with_bg:
            for j,threshold in enumerate(thresholds):
                
                for i in range(50): # TODO: adjust 50 if more background parts are used
                    torch_color = torch.zeros(1,3,1,1).to(image.device)
                    torch_color[0,0,0,0] = 204
                    torch_color[0,1,0,0] = 204
                    torch_color[0,2,0,0] = 204+i
                    color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                    #color_available_dilated = dilation1(color_available)

                    attribution_in_part = attribution * color_available
                    attribution_in_part = attribution_in_part.sum()# / (color_available_dilated.sum() + 1e-8)
                    
                    if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                        important_parts_for_thresholds[j].append('bg_' + str(i).zfill(3))


        return important_parts_for_thresholds

