# MICCAI 2020 Cerebrovascular Segmentation in MRA via Reverse Edge Attention Network
This repo is the official implementation of [Cerebrovascular Segmentation in MRA via Reverse Edge Attention Network](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_7)


<img width="893" alt="Net" src="https://user-images.githubusercontent.com/43660513/124488264-3cf57a00-dde2-11eb-914a-6c28cf7bcd6b.png">

## I. Experiment Results:
<img width="1189" alt="results" src="https://user-images.githubusercontent.com/43660513/124485668-652fa980-dddf-11eb-80c5-7391c95e8327.png">


## II. Usage:
Using the `train3d.py` and `predict3d.py` to train and test the model on your own dataset, respectively.

The proposed network model **RE-Net** is defined in the `model.py` in `models` folder.  It can be easily edited and embed in your own code.
## III. Requirements:
* PyTorch = 1.2.0
* tqdm
* SimpleITK
* visdom
## IV. Citation：
If our paper or code is helpful to you, please cite our paper. If you have any questions, please feel free to ask me.
```
@inproceedings{zhang2020cerebrovascular,
  title={Cerebrovascular Segmentation in MRA via Reverse Edge Attention Network},
  author={Zhang, Hao and Xia, Likun and Song, Ran and Yang, Jianlong and Hao, Huaying and Liu, Jiang and Zhao, Yitian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={66--75},
  year={2020},
  organization={Springer}
}
```


