import os
import io
import torch
import json
from PIL import Image, ImageDraw
import requests
from base64 import decodestring, decodebytes
import numpy as np
import random
import itertools

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RandomForeground(torch.nn.Module):
    """Randomly add foreground

    Args:
        
    """

    def __init__(self):
        super().__init__()
        
        self.min_parts = 0
        self.max_parts = 32
        self.min_size = 224/16
        self.max_size = 224/16
        self.colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to add foreground.

        Returns:
            PIL Image: Image to add foreground.
        """

        H,W = img.size
        
        nr_parts = random.randint(self.min_parts, self.max_parts)
        img1 = ImageDraw.Draw(img)  

        for i in range(nr_parts):
            size = random.randint(self.min_size, self.max_size)
            color = random.sample(self.colors,1)[0]
            coord_x = random.randint(0, W)
            coord_y = random.randint(0, H)
            shape = [(coord_x-size//2, coord_y-size//2), (coord_x+size//2, coord_y+size//2)]

            img1.rectangle(shape, fill = color, outline ="black")

       
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class FunnyBirds(Dataset):
    """FunnyBirds dataset."""

    def __init__(self, root_dir, mode, get_part_map=False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images. E.g. ./datasets/FunnyBirds
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.get_part_map = get_part_map
        self.transform = transform
        if transform != None:
            print('transforms only support image alternations')

        path_dataset_json = os.path.join(self.root_dir, 'dataset_' + self.mode + '.json')
        with open(path_dataset_json, 'r') as openfile:
            self.params = json.load(openfile)

        with open(os.path.join(self.root_dir, 'classes.json')) as f:
            self.classes = json.load(f)
        
        with open(os.path.join(self.root_dir, 'parts.json')) as f:
            self.parts = json.load(f)

        # this is the decoding from part_map to part
        self.colors_to_part = {(255,255,253): 'eye01',(255,255,254): 'eye02', (255,255,0): 'beak', (255,0,1): 'foot01', (255,0,2): 'foot02', (0,255,1): 'wing01', (0,255,2): 'wing02', (0,0,255): 'tail'}
        self.bg_color = (0,0,0)


    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        
        class_idx = self.params[idx]['class_idx']

        img_path = os.path.join(self.root_dir,
                                self.mode,
                                str(class_idx),
                                str(idx).zfill(6)+'.png')
        image = Image.open(img_path)
    
        image = transforms.ToTensor()(image)[:-1,:,:] # remove alpha
        if self.transform != None:
            image = self.transform(image)

        params = self.params[idx]

        if self.get_part_map:
            part_map_path = os.path.join(self.root_dir,
                                        self.mode + '_part_map',
                                        str(class_idx),
                                        str(idx).zfill(6)+'.png')
            part_map = Image.open(part_map_path)
            part_map = transforms.ToTensor()(part_map)[:-1,:,:] # remove alpha
            part_map = part_map * 255.
        else:
            part_map = 0

        sample = {'image': image, 'params': params, 'class_idx': class_idx, 'part_map': part_map, 'image_idx': idx}

        return sample
    
    #def add_foreground(img_tensor, coords=None, sizes=None, colors=None):

    
    def json_to_url(self, json, prefix = 'http://localhost:8081/render?', render_mode = 'default'):
        url = prefix
        url = url + 'render_mode=' + render_mode + '&'
        for key in list(json.keys()):
            if key == 'class_idx':
                continue
            url = url + key + '=' + str(json[key]) + '&'
        return url[:-1]


    def json_to_image_highres(self, json, render_mode='deault'):
        url = self.json_to_url(json, render_mode=render_mode)
        response = requests.get(url).content
        #image = Image.fromstring('RGB',(512,512),decodestring(response))
        image = decodebytes(response) 

        img = Image.open(io.BytesIO(image))

        return img

    def json_to_image(self, json):
        url = self.json_to_url(json)
        response = requests.get(url).content
        #image = Image.fromstring('RGB',(512,512),decodestring(response))
        image = decodebytes(response) 

        img = Image.open(io.BytesIO(image))
        newsize = (256, 256)
        img = img.resize(newsize)

        return img
    
    # params from dataloader have different format.
    # This function converts this format to format required for rendering

    # params are params loaded from the dataset
    # idx is the sample index of the batch
    def get_params_for_single(self, params, idx=0):
        out_params = {}
        for key in params.keys():
            if torch.is_tensor(params[key][idx]):
                out_params[key] = params[key][idx].item()
            else:
                out_params[key] = params[key][idx]
        return out_params

    def render(self, params, transform=None):
        while True:
            image = self.json_to_image(params)
            image = transforms.ToTensor()(image)[:-1,:,:] # remove alpha
            if not torch.all(image[0,:,:] == image[0,0,0]):
                image = image.unsqueeze(0)
                break
            
                  
        return image, params

    def render_highres(self, params, render_mode = 'default', transform=None):
        while True:
            image = self.json_to_image_highres(params, render_mode)
            image = transforms.ToTensor()(image)[:-1,:,:] # remove alpha
            if not torch.all(image[0,:,:] == image[0,0,0]):
                image = image.unsqueeze(0)
                break
            
                  
        return image, params

    def render_class(self, class_idx, scene_params, transform=None):
        
        params = scene_params #include camera, light, background

        for part in self.classes[class_idx]['parts'].keys():
            for part_attribute in self.parts[part][self.classes[class_idx]['parts'][part]]:
                new_key = part + '_' + part_attribute
                params[new_key] = self.parts[part][self.classes[class_idx]['parts'][part]][part_attribute]

        image, params = self.render(params, transform=transform)
                  
        return image, params

    def class_distance(self, class_idx_1, class_idx_2):

        class_1 = self.classes[class_idx_1]
        class_2 = self.classes[class_idx_2]
        distance = 0
        for key in class_1['parts'].keys():
            if class_1['parts'][key] != class_2['parts'][key]:
                distance += 1
        return distance

    # n is the distance between classes; n=1 means that one part is different
    def get_classes_with_distance_n(self, query_class_idx, n):
        

        classes_with_distance_n = []
        for single_class in self.classes:
            distance = self.class_distance(query_class_idx, single_class['class_idx'])
            if distance == n:
                classes_with_distance_n.append(single_class['class_idx'])
        return classes_with_distance_n


    def get_minimal_sufficient_part_sets(self, class_idx):
        class_info = self.classes[class_idx]
        
        keys_list = list(self.parts.keys())
        attributes_list = []
        
        attributes_subsets = []
        
        for key in keys_list:
            attributes_list.append(key + '_' + str(class_info['parts'][key]))
            
        for L in range(1, len(attributes_list)+1):
            for subset in itertools.combinations(attributes_list, L):
                attributes_subsets.append(list(subset))
                
        sufficient_attributes_subsets = []
        for subset in attributes_subsets:
            for i,other_class in enumerate(self.classes):
                if other_class['class_idx'] == class_idx:
                    continue
                attributes_list_other_class = []
                for key in keys_list:
                    attributes_list_other_class.append(key + '_' + str(other_class['parts'][key]))
                if set(subset) <= set(attributes_list_other_class):
                    break
                if i == len(self.classes)-1:

                    sufficient_attributes_subsets.append(subset)
                    
        minimal_sufficient_attributes_subsets = []
        for subset_current in sufficient_attributes_subsets:
            for i,subset in enumerate(sufficient_attributes_subsets):
                if set(subset_current) > set(subset):
                    break
                if i == len(sufficient_attributes_subsets)-1:
                    minimal_sufficient_attributes_subsets.append(subset_current)
        
        return minimal_sufficient_attributes_subsets

    # parts of form {beak}
    def get_classes_for_subset(self, parts):
        print(self.classes)
        print(self.parts)

    def single_params_to_part_idxs(self, params_single):
        parts_keys = list(self.parts.keys())
        parts_specification = {}
        for key in params_single.keys():
            part = key.split('_')[0]
            attribute = key.split('_')[1]
            if part in parts_keys:
                if part in parts_specification.keys():
                    parts_specification[part][attribute] = params_single[key]
                else:
                    parts_specification[part] = {}
                    parts_specification[part][attribute] = params_single[key]

        # parts_specification = {'beak': {'model': 'beak01.glb', 'color': 'yellow'}, 'eye': {'model': 'eye03.glb'}, 'foot': {'model': 'foot03.glb'}, 'tail': {'model': 'tail02.glb', 'color': 'red'}, 'wing': {'model': 'placeholder', 'color': 'placeholder'}}
        parts_specification_2 = {}
        for part in parts_specification.keys():
            try:
                idx = self.parts[part].index(parts_specification[part])
            except ValueError:
                idx = -1
            parts_specification_2[part] = idx
        # parts_specification_2 = {'beak': 0, 'eye': 2, 'foot': 2, 'tail': 1, 'wing': -1}
        return parts_specification_2
        


    def get_intervention(self, class_idx, image_idx, parts_removed):
        parts = list(self.parts.keys())

        keep_parts = list(set(parts) - set(parts_removed))
        image_name = 'body_' + '_'.join(sorted(keep_parts)) + '.png'
        path = os.path.join(self.root_dir, self.mode + '_interventions', str(class_idx), str(image_idx).zfill(6), image_name)

        sample = {}
        
        image = Image.open(path)
    
        image = transforms.ToTensor()(image)[:-1,:,:] # remove alpha
        if self.transform != None:
            image = self.transform(image)
        
        sample['image'] = image.unsqueeze(0)

        return sample

    def get_background_intervention(self, class_idx, image_idx, bg_object_id):

        image_name = str(bg_object_id) + '.png'            
        path = os.path.join(self.root_dir, self.mode + '_interventions', str(class_idx), str(image_idx).zfill(6), 'background_interventions', image_name)

        sample = {}
        
        image = Image.open(path)
    
        image = transforms.ToTensor()(image)[:-1,:,:] # remove alpha
        if self.transform != None:
            image = self.transform(image)
        
        sample['image'] = image.unsqueeze(0)

        return sample