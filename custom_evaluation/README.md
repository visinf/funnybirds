# Custom evaluations

## ProtoPNet

To run the ProtoPNet evaluation we need to render new images. Thus, please make sure that the rendering server is running on your machine (see [here](https://github.com/visinf/funnybirds/tree/main/render)).

You can download our ProtoPNet model and the corresponding /img folder (for prototype information) trained on FunnyBirds here:
```
cd /path/to/models/
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/protopnet_007_60push_final_1_checkpoint_best.pth.tar
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/img.zip
unzip img.zip
```

Unfortunately, we used an outdated model for the values reported in the paper. With the new model, the values with the corresponding calls look as follows (the conclusions do not change):

```python evaluate_protopnet.py --data /datasets/FunnyBirds --load_model_dir /path/to/models/ --epoch_number_str 60 --nr_itrs 100 --checkpoint_name /path/to/models/protopnet_007_60push_final_1_checkpoint_best.pth.tar```

nr_total_prototypes 1000
nr_reasonable_prototypes 628
Unreasonable prototypes: 1000 - 628 = 372

## Counterfactual Visual Explanations (CVE)

To run the CVE evaluation we need to render new images. Thus, please make sure that the rendering server is running on your machine (see [here](https://github.com/visinf/funnybirds/tree/main/render)).

You can download our VGG16 model trained on FunnyBirds here:
```
cd /path/to/models/
wget download.visinf.tu-darmstadt.de/data/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar
```

Unfortunately, we used an outdated model for the values reported in the paper. With the new model, the values with the corresponding calls look as follows (the conclusions do not change):

Same orientation and only one different part between classes
```python evaluate_cve.py --data /datasets/FunnyBirds --model vgg16 --checkpoint_name /path/to/models/vgg16_final_1_checkpoint_best.pth.tar --nr_itrs 100 --same_ori_one_diff```
Unreasonable swaps: 1.21

Different orientation and only one different part between classes
```python evaluate_cve.py --data /datasets/FunnyBirds --model vgg16 --checkpoint_name /path/to/models/vgg16_final_1_checkpoint_best.pth.tar --nr_itrs 100 --different_ori_one_diff```
Unreasonable swaps: 1.03

Same orientation and different number of different parts between classes
```python evaluate_cve.py --data /datasets/FunnyBirds --model vgg16 --checkpoint_name /path/to/models/vgg16_final_1_checkpoint_best.pth.tar --nr_itrs 100 --same_ori```
Unreasonable swaps: 3.04

Different orientation and different number of different parts between classes
```python evaluate_cve.py --data /datasets/FunnyBirds --model vgg16 --checkpoint_name /path/to/models/vgg16_final_1_checkpoint_best.pth.tar --nr_itrs 100```
Unreasonable swaps: 2.75