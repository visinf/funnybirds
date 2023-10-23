import json
import random
import math
import requests
from PIL import Image
from base64 import decodebytes
import io
import numpy as np
import os
from glob import glob
from shutil import rmtree
import argparse

def delete_rand_items(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

def create_classes_json(nr_classes, parts):
  
  unique_part_combinations = []
  classes = []
  part_keys = list(parts.keys())
  part_numbers = []
  for part in part_keys:
    part_numbers.append(len(parts[part]))

  i = 0
  while i < nr_classes:
    sample = {'class_idx': i, 'parts': {}}
    for p, part in enumerate(part_keys):
      part_number = part_numbers[p]
      random_part_idx = random.randint(0, part_number-1)
      
      sample['parts'][part] = random_part_idx
      if not sample['parts'] in unique_part_combinations:
        unique_part_combinations.append(sample['parts'])
        classes.append(sample)
        i += 1
  return classes

def create_dataset_json(samples_per_class, classes, parts, min_bg_parts, max_bg_parts, mode):
  dataset = []
  for c in range(len(classes)):
    current_class = classes[c]
    for s in range(samples_per_class):
      sample = {'class_idx': current_class['class_idx']}
      # parameter I need to set here:
      # http://localhost:8081/page?render_mode=default&camera_distance=700&camera_pitch=6.28&camera_roll=1.0&light_distance=300&light_pitch=6.0&light_roll=0.0
      # &beak_model=beak04.glb&beak_color=yellow&foot_model=foot01.glb&eye_model=eye02.glb&tail_model=tail01.glb&tail_color=red&wing_model=wing02.glb&wing_color=green
      # &bg_objects=0,1,2&bg_scale_x=20,20,20&bg_scale_y=20,20,20&bg_scale_z=20,20,20&bg_rot_x=20,2,3&bg_rot_y=1,5,100&bg_rot_z=1,2,100&bg_color=red,green,blue&bg_radius=100,150,200&bg_pitch=1.6,1,2&bg_roll=5,1.5,2.5
      sample['camera_distance'] = random.randint(200, 400)
      sample['camera_pitch'] = random.uniform(0, 2*math.pi)
      sample['camera_roll'] = random.uniform(0, 2*math.pi)
      sample['light_distance'] = 300
      sample['light_pitch'] = random.uniform(0, 2*math.pi)
      sample['light_roll'] = random.uniform(0, 2*math.pi)
      
      # set parts
      part_keys = list(parts.keys())
      if mode == 'train' or mode == 'train_part_map': # randomly remove n parts from the bird to allow interventions to be in domain
        if random.choice([0, 1]):
          nr_delete = random.randint(0, len(part_keys))
          part_keys_keep = delete_rand_items(part_keys, nr_delete)
        else:
          part_keys_keep = part_keys # removing from every sample parts reduces performance of trained networks quite a lot... so just remove parts from 50%
      else:
        part_keys_keep = part_keys

      for part in part_keys:
        part_instance = parts[part] [current_class['parts'][part]]
        for key in list(part_instance.keys()):
          value = part_instance[key]
          if part in part_keys_keep:
            sample[part + '_' + key] = value
          else:
            sample[part + '_' + key] = 'placeholder'

      # set background
      nr_bg_parts = random.randint(min_bg_parts, max_bg_parts-1)
      bg_objects = ''
      bg_radius = ''
      bg_pitch = ''
      bg_roll = ''
      bg_scale_x = ''
      bg_scale_y = ''
      bg_scale_z = ''
      bg_rot_x = ''
      bg_rot_y = ''
      bg_rot_z = ''
      bg_color = ''

      for bg_part_i in range(nr_bg_parts):
        bg_objects = bg_objects + str(random.randint(0, 4)) + ',' # TODO: 0-4 are the existing background part ids ADJUST IF NEW PART IS ADDED
        
        bg_radius = bg_radius + str(random.randint(100, 200)) + ','
        bg_pitch = bg_pitch + str(random.uniform(0, 2*math.pi)) + ','
        bg_roll = bg_roll + str(random.uniform(0, 2*math.pi)) + ','

        bg_scale_x = bg_scale_x + str(random.randint(5, 20)) + ','
        bg_scale_y = bg_scale_y + str(random.randint(5, 20)) + ','
        bg_scale_z = bg_scale_z + str(random.randint(5, 20)) + ','

        bg_rot_x = bg_rot_x + str(random.uniform(0, 2*math.pi)) + ','
        bg_rot_y = bg_rot_y + str(random.uniform(0, 2*math.pi)) + ','
        bg_rot_z = bg_rot_z + str(random.uniform(0, 2*math.pi)) + ','
        
        bg_color = bg_color + random.choice(['red', 'green', 'blue', 'yellow']) + ','
      
      sample['bg_objects'] = bg_objects

      sample['bg_radius'] = bg_radius
      sample['bg_pitch'] = bg_pitch
      sample['bg_roll'] = bg_roll

      sample['bg_scale_x'] = bg_scale_x
      sample['bg_scale_y'] = bg_scale_y
      sample['bg_scale_z'] = bg_scale_z

      sample['bg_rot_x'] = bg_rot_x
      sample['bg_rot_y'] = bg_rot_y
      sample['bg_rot_z'] = bg_rot_z

      sample['bg_color'] = bg_color

      dataset.append(sample)

  return dataset

def json_to_url(json, prefix = 'http://localhost:8081/render?', render_mode = 'default'):
  url = prefix
  url = url + 'render_mode=' + render_mode + '&'
  for key in list(json.keys()):
    if key == 'class_idx':
      continue
    url = url + key + '=' + str(json[key]) + '&'
  return url[:-1]
    


def json_to_image(json, mode):
  
  if mode == 'train' or mode == 'test':
    url = json_to_url(json)
  elif mode == 'train_part_map' or mode == 'test_part_map':
    url = json_to_url(json, render_mode='part_map')
  else:
    return NotImplementedError
  print(url)
  response = requests.get(url).content
  #image = Image.fromstring('RGB',(512,512),decodestring(response))
  image = decodebytes(response) 

  img = Image.open(io.BytesIO(image))
  newsize = (256, 256)
  if mode == 'train' or mode == 'test':
    img = img.resize(newsize)
  elif mode == 'train_part_map' or mode == 'test_part_map':
    img = img.resize(newsize, resample=Image.NEAREST)

  return img

  

def create_dataset(dataset_json, store_path, mode):
  for i,sample_json in enumerate(dataset_json):
    print(i)
    while True:
      img = json_to_image(sample_json, mode)
      # test if all values are the same
      im_matrix = np.array(img)
      if not np.all(im_matrix[:,:,0] == im_matrix[0,0,0]):
        path = os.path.join(store_path, str(sample_json['class_idx']))
        if not os.path.exists(path):
          os.makedirs(path)
        img.save(path + '/' + str(i).zfill(6) + '.png', 'png')
        #clean tmp dir
        pattern = os.path.join('/tmp', "puppeteer*")
        for item in glob(pattern):
          if not os.path.isdir(item):
              continue
          rmtree(item)
        break

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--mode', required=True,
                    choices=['train', 'train_part_map', 'test', 'test_part_map'],
                    help='Specify which data split you want to generate.')
parser.add_argument('--nr_classes', default=50, type=int,
                    help='The number of classes in the dataset.')
parser.add_argument('--nr_samples_per_class', default=10, type=int,
                    help='The number of samples per class.')
parser.add_argument('--root_path', required=True, type=str,
                    help='Path to the dataset. E.g. ./datasets')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed.')
parser.add_argument('--create_classes_json', action='store_true',
                    help='create_classes_json')
parser.add_argument('--create_dataset_json', action='store_true',
                    help='create_datasert_json')
parser.add_argument('--render_dataset', action='store_true',
                    help='create_datasert_json') 

args = parser.parse_args()

random.seed(args.seed)

#create directory
path = os.path.join(args.root_path, 'FunnyBirds')
if not os.path.exists(path):
    os.makedirs(path)

path_mode = os.path.join(args.root_path, 'FunnyBirds', args.mode)
if not os.path.exists(path_mode):
    os.makedirs(path_mode)


with open('parts.json') as f:
    parts = json.load(f)
print(parts)

if args.create_classes_json:
    classes = create_classes_json(args.nr_classes, parts)
    path_classes_json = os.path.join(path, 'classes.json')
    with open(path_classes_json, "w") as outfile:
        json.dump(classes, outfile)
    print('classes.json created')
else:
    path_classes_json = os.path.join(path, 'classes.json')
    with open(path_classes_json) as f:
        classes = json.load(f)
    print('classes.json loaded')

if args.create_dataset_json:
    dataset_json = create_dataset_json(args.nr_samples_per_class, classes, parts, 0, 35, args.mode)
    if args.mode == 'train' or args.mode == 'test': 
        path_dataset_json = os.path.join(path, 'dataset_' + args.mode + '.json')
        with open(path_dataset_json, "w") as outfile:
            json.dump(dataset_json, outfile)
    print('dataset_json created')
else:
    if args.mode == 'train' or args.mode == 'train_part_map':
      path_dataset_json = os.path.join(path, 'dataset_' + 'train' + '.json')
    elif args.mode == 'test' or args.mode == 'test_part_map':
      path_dataset_json = os.path.join(path, 'dataset_' + 'test' + '.json')

    with open(path_dataset_json) as f:
        dataset_json = json.load(f)
    print('dataset_json loaded')


if args.render_dataset:
    create_dataset(dataset_json, path_mode, args.mode)