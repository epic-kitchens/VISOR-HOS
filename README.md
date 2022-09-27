# VISOR - Hand Object Segmentation (HOS) Challenge 
## EPIC-KITCHENS VISOR Benchmark VIdeo Segmentations and Object Relations (NeurIPS 2022 - Datasets and Benchmarks Track)

Ahmad Darkhalil*, Dandan Shan*, Bin Zhu*, Jian Ma*, Amlan Kar, Richard Higgins, Sanja Fidler, David Fouhey, Dima Damen


[Project Webpage](https://epic-kitchens.github.io/VISOR/) / [Trailer](https://www.youtube.com/watch?v=yGodQAbYW_E) 
## Introduction
This repo contains code for the Hand-Object-Segmentation benchmarks and evaluations in EPCI-KITCHENS VISOR.

## Citing VISOR
When use this repo, any of our models or dataset, you need to cite the VISOR paper

```
@inproceedings{VISOR2022,
  title = {EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations},
  author = {Darkhalil, Ahmad and Shan, Dandan and Zhu, Bin and Ma, Jian and Kar, Amlan and Higgins, Richard and Fidler, Sanja and Fouhey, David and Damen, Dima},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  year = {2022}
}
```

## Environment

Conda environment recommended:
- cv2
- [pytorch](https://pytorch.org/get-started/locally/)
- [detectron2](https://github.com/facebookresearch/detectron2)
```
conda create --name hos
conda activate hos
pip install opencv-python
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
```

## Data Preparation

Download VISOR data from [EPIC-KITCHENS VISOR](https://epic-kitchens.github.io/VISOR/#downloads). Unzip it and rename it as `epick_visor`.

Generate a COCO format annotation of VISOR data for training:

&emsp;`--epick_visor`:the path to the annotation folder. 

&emsp;`--mode`: coco format data for different tasks, choose from `hos` or `active`.

&emsp;`--split`: generate for which split, choose from `train` and `val`.

&emsp;`--unzip_img`: only need to use this args once to unzip the orginally downloaded compressed images for each video. Worth noting that `unzip` command sometimes has some issue, which affects data loading later.

&emsp;`--copy_img`: copy images to get the same folder structure as in COCO.
```
python gen_coco_format.py \
--epick_visor_store=/path/to/epick_visor/GroundTruth-SparseAnnotations \
--mode=hos \
--split train val \
--unzip_img \
--copy_img \
``` 

Then the data structure looks like below:
```
datasets
├── epick_visor_coco_active
│   ├── annotations
│   │   ├── train.json
│   │   └── val.json
│   ├── train 
│   │   └── *.jpg
│   └── val 
│       └── *.jpg
└── epick_visor_coco_hos
    ├── annotations
    │   ├── train.json
    │   └── val.json
    ├── train 
    │   └── *.jpg
    └── val 
        └── *.jpg
```

Error Correction. In the script before generating the COCO version, we correct some errors first and save all jsons in `annotations_corrected` folder. 
- In the dataset, there are missing "on_which_hand" and "in_contact_object" labels for two images with gloves, `P06_13/P06_13_frame_0000000128.jpg` and `P06_13/P06_13_frame_0000000181.jpg`. We add the keys and values for them to make sure all images with gloves have these two keys.
- There is a typo on 13 images in train set where 'on_which_hand' is ['left hand', 'rigth hand'], we meant ['left hand', 'right hand'].
- Hand-object relations errors in 11 images.


Visualize the COCO version annotations from trainset:
```
python -m hos.data.datasets.epick ./datasets/epick_visor_coco_hos/annotations/train.json ./datasets/epick_visor_coco_hos/train epick_visor_2022_train
```


## Pre-trained Weights
Download our pre-trained weights into `checkpoints\` folder to run evaluation or demo code:
```
mkdir checkpoints && cd checkpoints
wget -O model_final_hos.pth https://www.dropbox.com/s/xi3249dbamv9wzs/model_final_hos.pth?dl=0
wget -O model_final_active.pth https://www.dropbox.com/s/jner6mn0hogmbav/model_final_active.pth?dl=0
cd ..
```


## Train
Hand and Contacted Object Segmentation (HOS) model:
```
python train_net_hos.py \
--config-file ./configs/hos/hos_pointrend_rcnn_R_50_FPN_1x.yaml \
--num-gpus 2 \
--dataset epick_hos  \
OUTPUT_DIR ./checkpoints/hos_train
```
Hand and Active Object Segmentation (Active) model:
```
python train_net_active.py \
--config-file ./configs/active/active_pointrend_rcnn_R_50_FPN_1x.yaml \
--num-gpus 2 \
--dataset epick_active \
OUTPUT_DIR ./checkpoints/active_train
```


## Evaluation
Hand and Contacted Object Segmentation (HOS) model:
```
python eval.py \
--config-file ./configs/hos/hos_pointrend_rcnn_R_50_FPN_1x.yaml \
--num-gpus 2 \
--eval-only \
OUTPUT_DIR ./checkpoints/hos \
MODEL.WEIGHTS ./checkpoints/model_final_hos.pth
```
Hand and Active Object Segmentation (Active) model:
```
python eval.py \
--config-file ./configs/active/active_pointrend_rcnn_R_50_FPN_1x.yaml \
--num-gpus 2 \
--eval-only \
OUTPUT_DIR ./checkpoints/active \
MODEL.WEIGHTS ./checkpoints/model_final_active.pth
```


## Demo
Create `inputs\` and `outputs\` folders, put images you want to test into `inputs\`:
```
mkdir inputs && mkdir outputs
```
Then run the demo:
```
python demo.py --inputs=inputs --outputs=outputs
```

