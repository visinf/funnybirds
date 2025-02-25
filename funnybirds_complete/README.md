# FunnyBirds Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository is an extended version of funnybirds-framework (https://github.com/visinf/funnybirds-framework/tree/main), containing all methods evaluated within the paper. 


[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. _ICCV_, 2023, **oral presentation**.

[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html) | [ArXiv](https://arxiv.org/abs/2308.06248) | [Video](https://www.youtube.com/watch?v=rOc-Wd4FN1E&t)

## Why multiple repositories?

This repository contains the code to replicate the complete FunnyBirds paper. As there are many different model architectures and explanation methods, and they all have specific requirements in their usage, this code might not be very easy to understand in a short time. For this reason, we leave funnybirds-framework as a starting point with more minimal examples. 

## What are the differences to funnybirds-framework?

### More Archtiectures
1. BagNet & BCos: 'bagnet33' and 'bcos_resnet50' are added as options in ```train.py``` and ```evaluate_explainability.py```. Note that BCos requires some additional changes for setting up the model and uses a different training loss. 

2. ProtoPNet: Training ProtoPNet follows a different pipeline than the rest of the models so the corresponding code can be found in ```models/protopnet/train_protopnet.py```.

For all three architectures, evaluation remains the same: You call ```evaluate_explainability.py``` with the architecture both as `--model` and `--explainer` argument. The functionality for each method is implemented in seperate classes in `explainers/explainer_wrapper.py`.

### More Evaluations
We add code for evaluating LIME and RISE as explanation methods in `evaluate_explainability.py`. These are compatible with ResNet, VGG16, and ViT and can be used the same way as the other standard methods, for instance:
```
DATA_DIR=/path/to/dataset/FunnyBirds/
MODEL_PATH=/path/to/models/
python evaluate_explainability.py --data $DATA_DIR --model vgg16 --explainer Rise --checkpoint_name $MODEL_PATH/vgg16_final_1_checkpoint_best.pth.tar --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence
```
For Rise, you first have to specify a path, where to save the occlusion masks (search for `maskspath` in `explainer_wrapper.py`).

The rest of the README is identical to the one in funnybirds-framework, explaining how to set up your conda environment, get our dataset, and generally work with the code.

## Getting Started

The following section provides a very detailed description of how to use the FunnyBirds framework. Even if the instructions might seem a bit long and intimidating, most of the steps are finished very quickly. So don't lose hope and if you have recommendations on how to improve the framework or the instructions, we would be grateful for your feedback.

### Download the dataset

The dataset requires ~1.6GB free disk space.
```
cd /path/to/dataset/
wget download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
unzip FunnyBirds.zip
rm FunnyBirds.zip
```

### Set up the environment

If you use conda you can create your environment as shown below:
```
conda create --name funnybirds-framework python=3.7
conda activate funnybirds-framework
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install captum -c pytorch
conda install -c conda-forge tqdm
conda install -c anaconda scipy
conda install conda-forge::einops=0.6.1
conda install anaconda::pillow
```

### Clone the repository

```
git clone https://github.com/visinf/funnybirds-framework.git
cd funnybirds-framework
```

After following the above steps, you can test if everything is properly set up by running:

```
DATA_DIR=/path/to/dataset/FunnyBirds/
python evaluate_explainability.py --data $DATA_DIR --model resnet50 --explainer InputXGradient --accuracy --gpu 0
```

This simply evaluates the accuracy of a randomly initialized ResNet-50 and should output something like 
```
Acc@1   0.00 (  1.00)
Acc@5   0.00 ( 10.00)
```
followed by an error (because all metrics must be evaluate to complete the script). If this is working, we can already continue with setting up the actual evaluation. In the FunnyBirds framework each method is a combination of a model and an explanation method.

### Prepare the model

If you want to evaluate a post-hoc explanation method on the standard models ResNet-50, VGG16, or ViT, you can simply download our model weights 
```
MODEL_PATH=/path/to/models/
cd $MODEL_PATH
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/resnet50_final_0_checkpoint_best.pth.tar
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/vit_base_patch16_224_final_1_checkpoint_best.pth.tar
```
and choose the models with the parameters ```--model [resnet50,vgg16, vit_b_16] --checkpoint_name $MODEL_PATH/model.pth.tar```. To verify this, running again
```
python evaluate_explainability.py --data $DATA_DIR --model resnet50 --checkpoint_name $MODEL_PATH/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --gpu 0
```
should now output an accuracy score close to 1.0. If you want to use your own model, you have to **train it** and **add it to the framework**.

#### Train a new model

For training your own model please use ```train.py```.

First enter your model name to the list of valid choices of the --model argument of the parser:
```
choices=['resnet50', 'vgg16', ...]
-->
choices=['resnet50', 'vgg16', ..., 'your_model']
```
Next, instantiate your model, load the ImageNet weights, and change the output dimension to 50, e.g.:
```python
# create model
if args.model == 'resnet50':
    model = resnet50(pretrained=args.pretrained)
    model.fc = torch.nn.Linear(2048, 50)
elif args.model == 'vgg16':
    model = vgg16(pretrained=args.pretrained)
    model.classifier[-1] = torch.nn.Linear(4096, 50)
elif ...
elif args.model == 'your_model':
    model = your_model()
    model.load_state_dict(torch.load('path/to/your/model_weights'))
    model.classifier[-1] = torch.nn.Linear(XXX, 50)
else:
    print('Model not implemented')
```

Now you can train your model by calling
```
python train.py --data $DATA_DIR --model your_model --checkpoint_dir $MODEL_PATH --checkpoint_prefix your_model --gpu 0 --multi_target --pretrained --seed 0
```
Don't forget to adjust the hyperparameters accordingly.

#### Add a new model to the framework

To add the model to the framework you first have to go to ```./models/modelwrapper.py``` and define a new class for your model that implements a ```forward()``` function and a ```load_state_dict()``` function (if ```StandardModel``` does not work for you). For examples you can refer to ```StandardModel``` or to the [complete FunnyBirds repository](https://github.com/visinf/funnybirds).
Next, you have to add the model to the choices list and the available models in ```evaluate_explainability.py``` as was done in [Train a new model](https://github.com/visinf/funnybirds-framework/tree/main#train-a-new-model).

### Prepare the explanation method

Each explanation method is wrapped in an explainer_wrapper that implements the interface functions and the function to generate the explanation:
```python
get_important_parts()
get_part_importance()
explain()
```

Currently, the code supports the explainers InputXGradient, Integrated Gradients, and the ViT specific methods Rollout and CheferLRP.

To implement your own wrapper, go to ```./explainers/explainer_wrapper.py``` and have a look at the ```CustomExplainer``` class. Here you can add your own explainer. If you want to evaluate an attribution method, simply let ```CustomExplainer``` inherit from ```AbstractAttributionExplainer``` and implement ```explain()``` and maybe ```__init__()```. If you want to evaluate another explanation type you also have to implement ```get_important_parts()``` and/or ```get_part_importance()```. For examples you can refer to the full [FunnyBirds repository](https://github.com/visinf/funnybirds) or the provided ```CaptumAttributionExplainer```.

The inputs and outputs of the interface functions ```get_part_importance()``` and ```get_important_parts()``` are defined as:

Inputs:
- ```image```: The input image. Torch tensor of size ```[1, 3, 256, 256]```.
- ```part_map```: The corresponding segmentation map where one color denotes one part. Torch tensor of size ```[1, 3, 256, 256]```.
- ```target```: The target class. Torch tensor of size ```[1]```.
- ```colors_to_part```: A list that maps colors to parts. Dictionary: ```{(255, 255, 253): 'eye01', (255, 255, 254): 'eye02', (255, 255, 0): 'beak', (255, 0, 1): 'foot01', (255, 0, 2): 'foot02', (0, 255, 1): 'wing01', (0, 255, 2): 'wing02', (0, 0, 255): 'tail'}```
- ```thresholds```: The different thresholds to use to estimate which parts are important. Numpy array of size ```(80,)```.
- ```with_bg```: Include the background parts in the computation. Boolean

Outputs:
- ```get_important_parts()``` A list with the same length as the number of thresholds (we get one result per threshold). Each inner list contains the strings of the parts that are estimated to be important by the explanation, e.g., ```['eye', 'beak', 'foot', 'wing', 'tail']```.
- ```get_part_importance()``` A dictionary with the part strings as keys and the estimated importances as value, e.g., ```{'eye': -0.040, 'beak': -1.25, 'foot': -0.504, 'wing': -0.501, 'tail': 0.3185}```.

Finally, you have to add your CustomExplainer to the ```evaluate_explainbility.py``` script by instantiating it in:
```python
elif args.explainer == 'CustomExplainer':
    ...
```

### Run the evaluation

If you have successfully followed all of the above steps you should be able to run the evaluation using the following command:
```
python evaluate_explainability.py --data $DATA_DIR --model your_model --checkpoint_name $MODEL_PATH/your_model_checkpoint_best.pth.tar --explainer CustomExplainer --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0
```
The evaluation for ResNet-50 with InputXGradient can be run with:
```
python evaluate_explainability.py --data $DATA_DIR --model resnet50 --checkpoint_name $MODEL_PATH/resnet50_final_0_checkpoint_best.pth.tar --explainer InputXGradient --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0
```

and should result in 

```
FINAL RESULTS:
Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS
0.998   0.7353  0.602   0.532   0.54372 0.99826 0.54592 0.806
Best threshold: 0.01620253164556962
```

#### Plot your results
To obtain the spider plots used in our paper, simply add your results to line 10 of plot_results.py and run the script. The result is stored in plot.png and should look like this:

<img src="https://github.com/visinf/funnybirds-framework/blob/main/resources/plot.png" width="256">

## Citation
If you find our work helpful, please consider citing
```
@inproceedings{Hesse:2023:FunnyBirds,
  title     = {Funny{B}irds: {A} Synthetic Vision Dataset for a Part-Based Analysis of Explainable {AI} Methods},
  author    = {Hesse, Robin and Schaub-Meyer, Simone and Roth, Stefan},
  booktitle = {2023 {IEEE/CVF} International Conference on Computer Vision (ICCV), Paris, France, October 2-6, 2023},
  year      = {2023},
  publisher = {{IEEE}}, 
  pages     = {3981-3991}
}
```



