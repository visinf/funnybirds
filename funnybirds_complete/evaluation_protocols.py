import torch
from tqdm import tqdm
from enum import Enum
from scipy import stats

from torch.utils.data import DataLoader

from datasets.funny_birds import FunnyBirds


def accuracy_protocol(model, args):

    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
            self.name = name
            self.fmt = fmt
            self.summary_type = summary_type
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)
        
        def summary(self):
            fmtstr = ''
            if self.summary_type is Summary.NONE:
                fmtstr = ''
            elif self.summary_type is Summary.AVERAGE:
                fmtstr = '{name} {avg:.3f}'
            elif self.summary_type is Summary.SUM:
                fmtstr = '{name} {sum:.3f}'
            elif self.summary_type is Summary.COUNT:
                fmtstr = '{name} {count:.3f}'
            else:
                raise ValueError('invalid summary type %r' % self.summary_type)
            
            return fmtstr.format(**self.__dict__)

    def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    transforms = None

    test_dataset = FunnyBirds(args.data, 'test', transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for samples in tqdm(test_loader):
        images = samples['image']
        target = samples['class_idx']
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

    print(top1)
    print(top5)

    return top1.avg.item() / 100

def controlled_synthetic_data_check_protocol(model, explainer, args):

    transforms = None

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    mcsdc_for_thresholds = {}
    for threshold in thresholds:
        mcsdc_for_thresholds[threshold] = 0
    number_valid_samples = 0
    for samples in tqdm(test_loader):
        images = samples['image']
        target = samples['class_idx']
        part_maps = samples['part_map']
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            part_maps = part_maps.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # make sure that model correctly classifies instance
        output = model(images)
        if output.argmax(1) != target:
            continue
         
        important_parts_for_thresholds = explainer.get_important_parts(images, part_maps, target, test_dataset.colors_to_part, thresholds=thresholds)
        for important_parts, threshold in zip(important_parts_for_thresholds, thresholds):
            minimal_sufficient_part_sets = test_dataset.get_minimal_sufficient_part_sets(target[0].item())
            max_J = 0
            for minimal_sufficient_part_set in minimal_sufficient_part_sets:
                minimal_sufficient_part_set = list(map(lambda part_string: ''.join((x for x in part_string if x.isalpha())), minimal_sufficient_part_set))
                J_current = len(set(minimal_sufficient_part_set).intersection(set(important_parts))) / len(minimal_sufficient_part_set)
                if J_current > max_J:
                    max_J = J_current


            mcsdc_for_thresholds[threshold] += max_J
        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        mcsdc_for_thresholds[threshold] = mcsdc_for_thresholds[threshold] / number_valid_samples

    print('mcsdcs: ', mcsdc_for_thresholds)

    return mcsdc_for_thresholds



def single_deletion_protocol(model, explainer, args):

    transforms = None

    # first get scores for different removed parts and original image
    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    correlations = []
    number_valid_samples = 0
    for sample in tqdm(test_loader):
        image = sample['image']
        target = sample['class_idx']
        part_map = sample['part_map']
        params = sample['params']
        class_idxs = sample['class_idx']
        image_idxs = sample['image_idx']
        params = test_dataset.get_params_for_single(params)
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            part_map = part_map.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        score = {}

        output = model(image)
        original_score = output[0,target].item()

        # get scores for removed parts
        bird_parts_keys = list(test_dataset.parts.keys())

        for remove_part in bird_parts_keys:
            image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), [remove_part])['image']

            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)
        
            score[remove_part.split('_')[0]] = output[0,target].item() #only keep part name, i.e. eye, instead of eye_model


        part_importances = explainer.get_part_importance(image, part_map, target, test_dataset.colors_to_part)
        score_diffs = {}
        for score_key in score.keys():
            score_diffs[score_key] = original_score - score[score_key]

        score_diffs_normalized = []
        part_importances_normalized = []
        for key in score_diffs.keys():
            score_diffs_normalized.append(score_diffs[key]) # not necessary to normalize with pearson coefficient
            part_importances_normalized.append(part_importances[key])

        correlation, p_value = stats.spearmanr(score_diffs_normalized, part_importances_normalized)

        import math
        if math.isnan(correlation):
            continue
        
        correlations.append(correlation * 0.5 + 0.5)


        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    print('Mean Single Deletion Correlation: ', sum(correlations)/len(correlations))
    return sum(correlations)/len(correlations)

def preservation_check_protocol(model, explainer, args):

    transforms = None

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []

    number_valid_samples = 0
    for samples in tqdm(test_loader):
        images = samples['image']
        target = samples['class_idx']
        part_maps = samples['part_map']
        params = samples['params']
        class_idxs = samples['class_idx']
        image_idxs = samples['image_idx']
        params = test_dataset.get_params_for_single(params)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            part_maps = part_maps.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        model_prediction_original = output.argmax(1)
        
        important_parts_for_thresholds = explainer.get_important_parts(images, part_maps, model_prediction_original, test_dataset.colors_to_part, thresholds=thresholds)
        for important_parts, threshold in zip(important_parts_for_thresholds, thresholds):
            all_parts = list(test_dataset.parts.keys())
            parts_removed = list(set(all_parts)-set(important_parts))

            image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), parts_removed)['image']
            image2 = image2.cuda(args.gpu, non_blocking=True)
            output2 = model(image2)
            model_prediction_removed = output2.argmax(1)

            if model_prediction_original == model_prediction_removed:
                scores_for_thresholds[threshold].append(1.)
            else:
                scores_for_thresholds[threshold].append(0.)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        scores_for_thresholds[threshold] = sum(scores_for_thresholds[threshold]) / len(scores_for_thresholds[threshold])
    
    print('Preservation Check Score: ', scores_for_thresholds)
    return scores_for_thresholds


def deletion_check_protocol(model, explainer, args):
    
    transforms = None

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []
        
    number_valid_samples = 0
    for samples in tqdm(test_loader):
        images = samples['image']
        part_maps = samples['part_map']
        class_idxs = samples['class_idx']
        image_idxs = samples['image_idx']

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            part_maps = part_maps.cuda(args.gpu, non_blocking=True)

        output = model(images)
        model_prediction_original = output.argmax(1)
        
        important_parts_for_thresholds = explainer.get_important_parts(images, part_maps, model_prediction_original, test_dataset.colors_to_part, thresholds=thresholds)
        for important_parts, threshold in zip(important_parts_for_thresholds, thresholds):
            parts_removed = important_parts

            image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), parts_removed)['image']
            image2 = image2.cuda(args.gpu, non_blocking=True)
            output2 = model(image2)
            model_prediction_removed = output2.argmax(1)

            if model_prediction_original == model_prediction_removed:
                scores_for_thresholds[threshold].append(0.)
            else:
                scores_for_thresholds[threshold].append(1.)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        scores_for_thresholds[threshold] = sum(scores_for_thresholds[threshold]) / len(scores_for_thresholds[threshold])
   
    print('Deletion Check Scores: ', scores_for_thresholds)
    return scores_for_thresholds


def target_sensitivity_protocol(model, explainer, args):
    
    def class_overlap(parts1, parts2):
        overlap_parts = []
        for key in parts1.keys():
            if parts1[key] == parts2[key]:
                overlap_parts.append(key)
        return overlap_parts
    
    transforms = None

    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    target_sensitivity_score = []
    number_valid_samples = 0
    number_assumption_wrong = 0

    assumption_strengths = []


    for sample in tqdm(test_loader):
        image = sample['image']
        target = sample['class_idx']
        part_map = sample['part_map']
        class_idxs = sample['class_idx']
        image_idxs = sample['image_idx']
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            part_map = part_map.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # make sure that model correctly classifies instance
        output = model(image)

        # get two classes that have each 2 parts in common with current target class
        classes_w_two_overlap = test_dataset.get_classes_with_distance_n(target[0], 3)
        #get two classes out of these that don't haver overlap in the two parts that overlap with target class. I.e. one overlaps in foot and beak and the other in tail and wing
        found_classes = False
        for class1_idx in range(len(classes_w_two_overlap)):
            for class2_idx in range(class1_idx+1, len(classes_w_two_overlap)):
                class1 = classes_w_two_overlap[class1_idx]
                class2 = classes_w_two_overlap[class2_idx]
                parts_target = test_dataset.classes[target[0]]['parts']
                parts_class1 = test_dataset.classes[class1]['parts']
                parts_class2 = test_dataset.classes[class2]['parts']
                overlap_target_class1 = class_overlap(parts_target, parts_class1)
                overlap_target_class2 = class_overlap(parts_target, parts_class2)
                
                if set(overlap_target_class1).isdisjoint(set(overlap_target_class2)):
                    found_classes = True
                    break
            if found_classes:
                break
                
        class1 = torch.tensor([class1]).cuda(args.gpu, non_blocking=True)
        class2 = torch.tensor([class2]).cuda(args.gpu, non_blocking=True)
        
        # skip sample if assumption does not hold
        # for class a: removing A parts should result in larger drop than removing B parts and removing B parts should result in larger increase than removing A parts (its the same)
        # for class b: removing B parts should result in larger drop than removing A parts 
  
        image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), overlap_target_class1)['image']


        image2 = image2.cuda(args.gpu, non_blocking=True)
        output_wo_parts_from_class1 = model(image2)

        image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), overlap_target_class2)['image']


        image2 = image2.cuda(args.gpu, non_blocking=True)
        output_wo_parts_from_class2 = model(image2)

        drop_class1_when_rm_class1_parts = output_wo_parts_from_class1[0][class1] - output[0][class1]
        drop_class1_when_rm_class2_parts = output_wo_parts_from_class2[0][class1] - output[0][class1]

        drop_class2_when_rm_class1_parts = output_wo_parts_from_class1[0][class2] - output[0][class2]
        drop_class2_when_rm_class2_parts = output_wo_parts_from_class2[0][class2] - output[0][class2]

        #smaller because the drop should be more negative
        if not (drop_class1_when_rm_class1_parts < drop_class1_when_rm_class2_parts and drop_class2_when_rm_class2_parts < drop_class2_when_rm_class1_parts):
            number_assumption_wrong += 1
            continue

        assumption_strengths.append(drop_class1_when_rm_class2_parts.item() - drop_class1_when_rm_class1_parts.item())

        part_importances_class1 = explainer.get_part_importance(image, part_map, class1, test_dataset.colors_to_part)
        part_importances_class2 = explainer.get_part_importance(image, part_map, class2, test_dataset.colors_to_part)

        overlap_target_class1_importance_class1 = 0
        overlap_target_class1_importance_class2 = 0
        for part in overlap_target_class1:
            overlap_target_class1_importance_class1 += part_importances_class1[part]
            overlap_target_class1_importance_class2 += part_importances_class2[part]

        if overlap_target_class1_importance_class1 > overlap_target_class1_importance_class2:
            target_sensitivity_score.append(1.)
        else:
            target_sensitivity_score.append(0.)


        overlap_target_class2_importance_class1 = 0
        overlap_target_class2_importance_class2 = 0
        for part in overlap_target_class2:
            overlap_target_class2_importance_class1 += part_importances_class1[part]
            overlap_target_class2_importance_class2 += part_importances_class2[part]

        if overlap_target_class2_importance_class1 < overlap_target_class2_importance_class2:
            target_sensitivity_score.append(1.)
        else:
            target_sensitivity_score.append(0.)

        number_valid_samples += 1
        if args.nr_itrs == number_valid_samples:
            break

    target_sensitivity_score = sum(target_sensitivity_score) / len(target_sensitivity_score)
    print('Number of filtered samples:', number_assumption_wrong)
    print('Number of valid samples:', number_valid_samples)
    print('Target Sensitivity Score: ', target_sensitivity_score)
    print('Assumption Strength: ', sum(assumption_strengths) / len(assumption_strengths))
    return target_sensitivity_score

def distractibility_protocol(model, explainer, args):
    transforms = None

    # first get scores for different removed parts and original image
    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    thresholds = explainer.get_p_thresholds()
    scores_for_thresholds = {}
    for threshold in thresholds:
        scores_for_thresholds[threshold] = []


    number_valid_samples = 0
    for sample in tqdm(test_loader):
        image = sample['image']
        target = sample['class_idx']
        part_map = sample['part_map']
        params = sample['params']
        class_idxs = sample['class_idx']
        image_idxs = sample['image_idx']
        params = test_dataset.get_params_for_single(params)
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            part_map = part_map.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        score = {}

        output = model(image)
        original_score = output[0,target].item()

        bird_parts_keys = list(test_dataset.parts.keys())

        for remove_part in bird_parts_keys:
            image2 = test_dataset.get_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), [remove_part])['image']

            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)
        
            score[remove_part.split('_')[0]] = output[0,target].item() #only keep part name, i.e. eye, instead of eye_model
        
        bg_keys = list(filter(lambda x: x.startswith('bg_'), params.keys()))
        bg_object_ids = [int(s) for s in re.findall(r'\b\d+\b', params[bg_keys[0]])]

        for i in range(len(bg_object_ids)):
            image2 = test_dataset.get_background_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), i)['image']

            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)
        
            score['bg_' + str(i).zfill(3)] = output[0,target].item()

        threshold_for_bg_importances = original_score * 0.05 # 5%
        irrelevant_parts = []

        for score_key in score.keys():
            score_diff = original_score - score[score_key]
            
            if abs(score_diff) < abs(threshold_for_bg_importances) or original_score < score[score_key]: # second condition: removing the part increases the class evidence --> part is not important
                irrelevant_parts.append(score_key)

        if len(irrelevant_parts) == 0:
            print('There are no irrelevant parts')
            continue

        explanation_important_parts_for_thresholds = explainer.get_important_parts(image, part_map, target, test_dataset.colors_to_part, with_bg=True, thresholds=thresholds)
        for explanation_important_parts, threshold in zip(explanation_important_parts_for_thresholds, thresholds):
            J_current = len(set(explanation_important_parts).intersection(set(irrelevant_parts))) / len(irrelevant_parts)

            scores_for_thresholds[threshold].append(J_current)

        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    for threshold in thresholds:
        scores_for_thresholds[threshold] = 1 - sum(scores_for_thresholds[threshold]) / (len(scores_for_thresholds[threshold]) + 1e-8)  

    print('Mean Distractibility Scores: ', scores_for_thresholds)
    return scores_for_thresholds


import re
def background_independence_protocol(model, args):

    transforms = None

    # first get scores for different removed parts and original image
    test_dataset = FunnyBirds(args.data, 'test', get_part_map=True, transform = transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    total_background_parts = 0
    number_relevant_background_parts = 0

    number_valid_samples = 0
    for sample in tqdm(test_loader):
        image = sample['image']
        target = sample['class_idx']
        params = sample['params']
        class_idxs = sample['class_idx']
        image_idxs = sample['image_idx']

        params = test_dataset.get_params_for_single(params)
        if args.gpu is not None:
            image = image.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        score = {}
        output = model(image)
        original_score = output[0,target].item()
        bg_keys = list(filter(lambda x: x.startswith('bg_'), params.keys()))
        bg_object_ids = [int(s) for s in re.findall(r'\b\d+\b', params[bg_keys[0]])]

        for i in range(len(bg_object_ids)):
            image2 = test_dataset.get_background_intervention(class_idxs.squeeze(0).item(), image_idxs.squeeze(0).item(), i)['image']
            image2 = image2.cuda(args.gpu, non_blocking=True)
            output = model(image2)
            score['bg_' + str(i).zfill(3)] = output[0,target].item()       

        threshold_for_bg_importances = original_score * 0.05 # 5%

        for score_key in score.keys():
            score_diff = original_score - score[score_key]
            total_background_parts += 1.
            if abs(score_diff) >= abs(threshold_for_bg_importances) and original_score > score[score_key]:
                number_relevant_background_parts += 1.
  
        number_valid_samples += 1

        if args.nr_itrs == number_valid_samples:
            break

    background_dependence = 1 - number_relevant_background_parts/total_background_parts

    print('Background Dependence Score: ', background_dependence)
    return background_dependence
