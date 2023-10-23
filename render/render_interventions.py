import json
import random
import requests
from PIL import Image
from base64 import decodebytes
import io
import numpy as np
import os
from glob import glob
from shutil import rmtree
import argparse


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

  
import itertools

def render_interventions(dataset_json, store_path, parts, mode):

  # all part combinations
  part_keys = list(parts.keys())
  part_combinations = []
  for L in range(len(part_keys) + 1):
    for subset in itertools.combinations(part_keys, L):
        part_combinations.append(list(subset))

  for i,sample_json in list(reversed(list(enumerate(dataset_json))))[199:]:
  #for i,sample_json in list(enumerate(dataset_json))[2299:]:
    print(i)
    
    path_class = os.path.join(store_path, str(sample_json['class_idx']))
    if not os.path.exists(path_class):
      os.makedirs(path_class)

    path_sample = os.path.join(store_path, str(sample_json['class_idx']), str(i).zfill(6))
    if not os.path.exists(path_sample):
      os.makedirs(path_sample)

    for part_combination in part_combinations:
      keep_parts = part_combination
      remove_parts = list(set(part_keys) - set(keep_parts))
            

      sample_json_for_interventions = sample_json.copy()

      for part in remove_parts:
        sample_json_for_interventions[part + '_model'] = ''

      if os.path.isfile(path_sample + '/body_' + '_'.join(sorted(keep_parts)) + '.png'):
        print(path_sample + '/body_' + '_'.join(sorted(keep_parts)) + '.png', 'already exists')
        continue

      while True:
        img = json_to_image(sample_json_for_interventions, mode)
        im_matrix = np.array(img)
        if not np.all(im_matrix[:,:,0] == im_matrix[0,0,0]):
          print('SAVE IMAGE')
          img.save(path_sample + '/body_' + '_'.join(sorted(keep_parts)) + '.png', 'png')
          #clean tmp dir
          pattern = os.path.join('/tmp', "puppeteer*")
          for item in glob(pattern):
            if not os.path.isdir(item):
                continue
            rmtree(item)
          break



import re
# render background interventions where at most one part is removed
def render_background_interventions(dataset_json, store_path, parts, mode):
  print('render bg interventions')
  for i,sample_json in enumerate(dataset_json):
    #if i != 2:
    #  continue 
    print(i)

    path_bg = os.path.join(store_path, str(sample_json['class_idx']), str(i).zfill(6), 'background_interventions')
    if not os.path.exists(path_bg):
      os.makedirs(path_bg)
    bg_keys = list(filter(lambda x: x.startswith('bg_'), sample_json.keys()))
    bg_objects = re.findall(r'\d+', sample_json[bg_keys[0]])
    for n in range(len(bg_objects)): # iterate over N bg parts
      sample_json_for_interventions = sample_json.copy()

      for bg_key in bg_keys:
          print(bg_key)
          vals = sample_json_for_interventions[bg_key].split(',')
          del vals[n]
          vals = ','.join(vals)
          sample_json_for_interventions[bg_key] = vals

      path_sample = path_bg + '/' + str(n) + '.png'

      if os.path.isfile(path_sample):
        print(path_sample, 'already exists')
        continue

      while True:
        img = json_to_image(sample_json_for_interventions, mode)
        im_matrix = np.array(img)
        if not np.all(im_matrix[:,:,0] == im_matrix[0,0,0]):
          img.save(path_sample, 'png')
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
parser.add_argument('--root_path', required=True, type=str,
                    help='Path to the dataset. E.g. ./datasets')
parser.add_argument('--seed', default=0, type=int,
                    help='Random seed.')

args = parser.parse_args()

random.seed(args.seed)

#create directory
path = os.path.join(args.root_path, 'FunnyBirds')
if not os.path.exists(path):
    os.makedirs(path)

path_mode = os.path.join(args.root_path, 'FunnyBirds', args.mode + '_interventions')
if not os.path.exists(path_mode):
    os.makedirs(path_mode)


with open('parts.json') as f:
    parts = json.load(f)
print(parts)


path_classes_json = os.path.join(path, 'classes.json')
with open(path_classes_json) as f:
    classes = json.load(f)
print('classes.json loaded')

if args.mode == 'train' or args.mode == 'train_part_map':
  path_dataset_json = os.path.join(path, 'dataset_' + 'train' + '.json')
elif args.mode == 'test' or args.mode == 'test_part_map':
  path_dataset_json = os.path.join(path, 'dataset_' + 'test' + '.json')

with open(path_dataset_json) as f:
    dataset_json = json.load(f)
print('dataset_json loaded')


render_interventions(dataset_json, path_mode, parts, args.mode)
render_background_interventions(dataset_json, path_mode, parts, args.mode)
