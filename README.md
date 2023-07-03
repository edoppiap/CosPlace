
# Image Geo-Localization Classification based on CosPlace Method

This is the project for Machine Learning and Deep Learning course at Politecnico di Torino, 2022-2023. This project use a provided dataset called San Francisco eXtra Small and an evaluation dataset called Tokyo eXtra Small (SF_XS, go [here](https://drive.google.com/drive/folders/1WzSLnv05FLm-XqP5DxR5nXaaixH23uvV?usp=sharing) to download both the datasets), and a highly scalable training method (called CosPlace), which allows to reach SOTA results with compact descriptors.

## Authors

* Gianluca Guzzetta s308449 [GitHub](https://github.com/gguzzy)
* Martina Martini s306163 [GitHub](https://github.com/s261026)
* Emanuele Pietropaolo s319501 [GitHub](https://github.com/edoppiap)

## Train
After downloading the SF_XS dataset, simply run 

`$ python3 train.py --dataset_folder path/to/sf-xs`

the script automatically splits SF_XS in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 is used, which fits in less than 4GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run 

`$ python3 train.py --dataset_folder path/to/sf-xs --backbone efficientnet_v2_s --fc_output_dim 128`

You can also speed up your training with Automatic Mixed Precision (note that all results/statistics from the paper did not use AMP)

`$ python3 train.py --dataset_folder path/to/sf-xs --use_amp16`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

You can also try some autoaugmentations, simply running

`$ python3 train.py --dataset_folder path/to/sf-xs --autoaugment_policy IMAGENET`

#### Reproducibility
Results from the paper are fully reproducible, and we followed deep learning's best practices (average over multiple runs for the main results, validation/early stopping and hyperparameter search on the val set).
If you are a researcher comparing your work against ours, please make sure to follow these best practices and avoid picking the best model on the test set.

## Test
You can test a trained model as such

`$ python3 eval.py --dataset_folder path/to/sf-xl/processed --backbone VGG16 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Trained Models

We now have all our trained models on [PyTorch Hub](https://pytorch.org/docs/stable/hub.html), so that you can use them in any codebase without cloning this repository simply like this
```
import torch
model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
```

As an alternative, you can download the trained models from the table below, which provides links to models with different backbones and dimensionality of descriptors, trained on SF-XL.

<table>
  <tr>
    <th rowspan=2>Model</th>
    <th colspan=7>Dimension of Descriptors</th>
  </tr>
  <tr>
    <td>32</td>
    <td>64</td>
    <td>128</td>
    <td>256</td>
    <td>512</td>
    <td>1024</td>
    <td>2048</td>
  </tr>
  <tr>
    <td>ResNet-18</td>
    <td><a href="https://drive.google.com/file/d/1tfT8r2fBeMVAEHg2bVfCql5pV9YzK620/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-d_Yi3ly3bY6hUW1F9w144FFKsZtYBL4/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1HaQjGY5x--Ok0RcspVVjZ0bwrAVmBvrZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1hjkogugTsHTQ6GTuW3MHqx-t4cXqx0uo/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rQAC2ZddDjzwB2OVqAcNgCFEf3gLNa9U/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td><a href="https://drive.google.com/file/d/18AxbLO66CO0kG05-1YrRb1YwqN7Wgp6Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F2WMt7vMUqXBjsZDIwSga3N0l0r9NP2s/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/14U3jsoNEWC-QsINoVCWZaHFUGE20fIgZ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1Q2sZPEJfHAe19JaZkdgeFotUYwKbV_x2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1LgDaxCjbQqQWuk5qrPogfg7oN8Ksl1jh/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1VBLUiQJfmnZ4kVQIrXBW-AE1dZ3EnMv2/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1yNzxsMg34KO04UJ49ncANdCIWlB3aUGA/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-101</td>
    <td><a href="https://drive.google.com/file/d/1a5FqhujOn0Pr6duKrRknoOgz8L8ckDSE/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/17C8jBQluxsbI9d8Bzf67b5OsauOJAIuX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1w37AztnIyGVklBMtm-lwkajb0DWbYhhc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1G5_I4vX4s4_oiAC3EWbrCyXrCOkV8Bbs/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1uBKpNfMBt6sLIjCGfH6Orx9eQdQgN-8Z/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/12BU8BgfqFYzGLXXNaKLpaAzTHuN5I9gQ/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1PF7lsSw1sFMh-Bl_xwO74fM1InyYy1t8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>ResNet-152</td>
    <td><a href="https://drive.google.com/file/d/12pI1FToqKKt8I6-802CHWXDP-JmHEFSW/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1rTjlv_pNtXgxY8VELiGYvLcgXiRa2zqB/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1q5-szPBn4zL8evWmYT04wFaKjen66mrk/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1sCQMA_rsIjmD-f381I0f2yDf0At4TnSx/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1ggNYQfGSfE-dciKCS_6SKeQT76O0OXPX/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/15vBWuHVqEMxkAWWrc7IrkGsQroC65tPc/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1AlF7xPSswDLA1TdhZ9yTVBkfRnJm0Hn8/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td>VGG-16</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1YJTBwagC0v50oPydpKtsTnGZnaYOV0z-/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1vgw509lGBfJR46cGDJGkFcdBTGhIeyAH/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1-4JtACE47rkXXSAlRBFIbydimfKemdo7/view?usp=sharing">link</a></td>
    <td><a href="https://drive.google.com/file/d/1F6CT-rnAGTTexdpLoQYncn-ooqzJe6wf/view?usp=sharing">link</a></td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

Or you can download all models at once at [this link](https://drive.google.com/drive/folders/1WzSLnv05FLm-XqP5DxR5nXaaixH23uvV?usp=sharing)

## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CosFace implementation in PyTorch](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py)
- [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (for the GeM layer)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) (for the evaluation / test code)

```
