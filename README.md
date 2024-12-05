# FunnyBirds

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[R. Hesse](https://robinhesse.github.io/), [S. Schaub-Meyer](https://schaubsi.github.io/), and [S. Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp). **FunnyBirds: A Synthetic Vision Dataset for a Part-Based Analysis of Explainable AI Methods**. _ICCV_, 2023, **oral presentation**.

[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hesse_FunnyBirds_A_Synthetic_Vision_Dataset_for_a_Part-Based_Analysis_of_ICCV_2023_paper.html) | [ArXiv](https://arxiv.org/abs/2308.06248) | [Video](https://www.youtube.com/watch?v=rOc-Wd4FN1E&t)

> :warning: **Disclaimer**: This repository will contain the full code to reproduce our work, including the dataset rendering, the custom evaluations, and the framework code for all methods (code coming soon). As this code can be a bit unclear, we additionally provide a separat [FunnyBirds framework repository](https://github.com/visinf/funnybirds-framework) that provides the minimal working code to run your own evaluations on the FunnyBirds framework. If you just want to run the framework evaluation, we thus recommend using the FunnyBirds framework repository.

News: 
- The [FunnyBirds framework repository](https://github.com/visinf/funnybirds-framework) is ready to evaluate your own methods (September 13, 2023)
- You can now render the dataset yourself (October 23, 2023)
- We have added the code for the custom evaluations (February 14, 2024)

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
conda install -c conda-forge nodejs
pip install opencv-contrib-python (conda install -c conda-forge opencv messes the environment up) 
```

## Dataset Generation

To render the FunnyBirds dataset, please refer to the [render folder](https://github.com/visinf/funnybirds/tree/main/render).

## Custom Evaluations

To run the custom evaluations, please refer to the [custom_evaluation folder](https://github.com/visinf/funnybirds/tree/main/custom_evaluation).

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
