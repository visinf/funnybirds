import torch
import torch.nn as nn
from abc import abstractmethod


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
    
    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)


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
        part_importances = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)

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

        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1,3,1,1).to(image.device)
            torch_color[0,0,0,0] = part_color[0]
            torch_color[0,1,0,0] = part_color[1]
            torch_color[0,2,0,0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
            
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50):
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = 204
                torch_color[0,1,0,0] = 204
                torch_color[0,2,0,0] = 204+i
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                attribution_in_part = attribution_in_part.sum()
                
                bg_string = 'bg_' + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)


import cv2
from PIL import Image
class CaptumAttributionExplainer(AbstractAttributionExplainer):
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def explain(self, input, target=None, baseline=None):
        if self.explainer_name == 'Saliency' or self.explainer_name == 'InputXGradient' or self.explainer_name == 'DeepLift': 
            return self.explainer.attribute(input, target=target)

        elif self.explainer_name == 'LayerGradCam': 
            B,C,H,W = input.shape
            attr = self.explainer.attribute(input, target=target, relu_attributions=True)
            m = transforms.Resize((H,W), interpolation=Image.NEAREST)
            attr = m(attr)

            return attr

        elif self.explainer_name == 'IntegratedGradients':
            return self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50)


class IntegratedGradientsAbsoluteExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def explain(self, input, target=None, baseline=None):
        return self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50).abs()


from models.bcos.data_transforms import AddInverse
from torch.autograd import Variable
from models.bcos.utils import explanation_mode


class BcosExplainer(AbstractAttributionExplainer):
    
    def __init__(self, model):
        """
        An explainer for bcos explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model

    def explain(self, input, target):
        explanation_mode(self.model, True)
        _input = Variable(AddInverse()(input), requires_grad=True)        
        output = self.model.model(_input) # if directly using self.model it returns None
        target = output[0][target]
        _input.grad = None
        target[0].backward(retain_graph=True)
        w1 = _input.grad
        attribution = w1 * _input
        explanation_mode(self.model, False)
        return attribution

import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class LimeExplainer(AbstractAttributionExplainer):
    """
    A wrapper for LIME.
    https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    Args:
        model: PyTorch model.
    """
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        from lime import lime_image
        self.explainer = lime_image.LimeImageExplainer()
        self.model = model

    def batch_predict(self, images):
        self.model.eval()

        batch = torch.stack(tuple(i.float() for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(device)
        
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()

    def explain(self, input, target=None):
        # convert input into numpy
        assert input.shape[0] == 1
        dev = input.device
        img = input[0].permute(1,2,0).detach().cpu().numpy().astype(np.double)
        # explain
        explanation = self.explainer.explain_instance(img, 
                                        self.batch_predict, # classification function
                                        labels=range(50),
                                        top_labels=None,
                                        hide_color=0,
                                        num_samples=1000) # number of images that will be sent to classification function 
        temp, mask = explanation.get_image_and_mask(int(np.array(target[0].cpu())), positive_only=True, num_features=5, hide_rest=False)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        return mask.to(dev)

    # def get_part_importance(self, image, part_map, target, colors_to_part):
    #     return NotImplementedError

    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg = False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1 # B = 1

        img = image[0].permute(1,2,0).detach().cpu().numpy().astype(np.double)
        # explain
        explanation = self.explainer.explain_instance(img, 
                                        self.batch_predict, # classification function
                                        labels=range(50),
                                        top_labels=None,
                                        hide_color=0, 
                                        num_samples=1000) # number of images that will be sent to classification function
        temp, mask = explanation.get_image_and_mask(int(np.array(target[0].cpu())), positive_only=True, num_features=5, hide_rest=False)
        mask1 = mask.copy()
        for i in range(len(mask)):
            for j in range(len(mask1[i])):
                if mask1[j][i] > 0:
                    mask1[j][i] = 1
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        attribution = mask
        attribution = attribution.to(image.device)
        
        important_parts_for_thresholds = []

        for threshold in thresholds:
            important_parts = []
            for part_color in colors_to_part.keys():
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = part_color[0]
                torch_color[0,1,0,0] = part_color[1]
                torch_color[0,2,0,0] = part_color[2]
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()

                attribution_in_part = attribution * color_available
                attribution_in_part = attribution_in_part.sum()
                
                if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                    important_parts.append(colors_to_part[part_color])

            important_parts = list(map(lambda part_string: ''.join((x for x in part_string if not x.isdigit())), important_parts)) # remove 01 and 02 from parts
            important_parts = list(dict.fromkeys(important_parts)) # remove duplicates, e.g. feet, feet
            important_parts_for_thresholds.append(important_parts)
        

        if with_bg:
            for j,threshold in enumerate(thresholds):
                
                for i in range(50):
                    torch_color = torch.zeros(1,3,1,1).to(image.device)
                    torch_color[0,0,0,0] = 204
                    torch_color[0,1,0,0] = 204
                    torch_color[0,2,0,0] = 204+i
                    color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()

                    attribution_in_part = attribution * color_available
                    attribution_in_part = attribution_in_part.sum()
                    
                    if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                        important_parts_for_thresholds[j].append('bg_' + str(i).zfill(3))

        return important_parts_for_thresholds

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)


import os 
class RiseExplainer(AbstractAttributionExplainer):
    """
    A wrapper for RISE.
    Args:
        model: PyTorch model.
    """
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        from .rise import RISE
        self.explainer = RISE(model, (256, 256), 16)
        # Generate masks for RISE or use the saved ones.
        maskspath = '/path/to/masks.npy'
        generate_new = False

        if generate_new or not os.path.isfile(maskspath):
            self.explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
            print('Masks are generated.')
        else:
            self.explainer.load_masks(maskspath, p1=0.1)
            print('Masks are loaded.')

    def explain(self, input, target=None):
        assert input.shape[0] == 1
        attribution = self.explainer(input)
        attribution = attribution[target[0].int()].unsqueeze(0).unsqueeze(0)
        return attribution



from models.ViT.ViT_explanation_generator import Baselines, LRP
class ViTGradCamExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_cam_attn(input_, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution
    
class ViTRolloutExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_rollout(input_, start_layer=1).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

class ViTCheferLRPExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = LRP(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1, method="transformer_attribution").reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution
    
from models.bagnets.utils import generate_heatmap_pytorch
class BagNetExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model

    def explain(self, input, target):
        assert input.shape[0] == 1
        attribution_numpy = generate_heatmap_pytorch(self.model, input.cpu(), target, 33)
        attribution = torch.from_numpy(attribution_numpy).unsqueeze(0).unsqueeze(0).to(input.device)
        return attribution


import matplotlib.pyplot as plt
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
        self.load_model_dir = model.load_model_dir

        self.load_img_dir = os.path.join(self.load_model_dir, 'img')
        self.epoch_number_str = model.epoch_number_str
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
        B,C,H,W = image.shape
        idx = 0

        logits, min_distances = self.model(image, return_min_distances = True)
        conv_output, distances = self.model.model.push_forward(image)
        prototype_activations = self.model.model.distance_2_similarity(min_distances)
        prototype_activation_patterns = self.model.model.distance_2_similarity(distances)
            
        topk_classes = target
        c = topk_classes[0]

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

        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1,3,1,1).to(image.device)
            torch_color[0,0,0,0] = part_color[0]
            torch_color[0,1,0,0] = part_color[1]
            torch_color[0,2,0,0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
            
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50):
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = 204
                torch_color[0,1,0,0] = 204
                torch_color[0,2,0,0] = 204+i
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
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
                attribution_in_part = attribution * color_available
                attribution_in_part = attribution_in_part.sum()
                
                if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                    important_parts.append(colors_to_part[part_color])

            important_parts = list(map(lambda part_string: ''.join((x for x in part_string if not x.isdigit())), important_parts)) # remove 01 and 02 from parts
            important_parts = list(dict.fromkeys(important_parts)) # remove duplicates, e.g. feet, feet
            important_parts_for_thresholds.append(important_parts)

        if with_bg:
            for j,threshold in enumerate(thresholds):
                
                for i in range(50):
                    torch_color = torch.zeros(1,3,1,1).to(image.device)
                    torch_color[0,0,0,0] = 204
                    torch_color[0,1,0,0] = 204
                    torch_color[0,2,0,0] = 204+i
                    color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()

                    attribution_in_part = attribution * color_available
                    attribution_in_part = attribution_in_part.sum()
                    
                    if attribution_in_part > threshold * color_available.sum(): #threshold to decide how big attribution in part should be
                        important_parts_for_thresholds[j].append('bg_' + str(i).zfill(3))

        return important_parts_for_thresholds
    
    
class CustomExplainer(AbstractExplainer):

    def explain(self, input):
        return 0
    
    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg = False):
        return 0
    
    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg = False):
        return 0

    # if not inheriting from AbstractExplainer you need to add this function to your class as well
    #def get_p_thresholds(self):
    #    return np.linspace(0.01, 0.50, num=80)
