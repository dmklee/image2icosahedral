*Image to Icosahedral Projection for SO(3) Object Reasoning from Single-View Images*
---------------------------------------------------------------------
[Paper](https://arxiv.org/abs/2207.08925) | [Project Page](https://dmklee.github.io/image2icosahedral/)

---------------------------------------------------------------------
This is the code for "Image to Icosahedral Projection for SO(3) Object 
Reasoning from Single-View Images" presented at NeurReps Workshop at NeurIPS 2022 and to
appear in PMLR Volume on Symmetry and Geometry.

## Installation
The code was tested using Python 3.8, and the neural networks are instantiated with PyTorch.
```
pip install -r requirements.txt
```

## Downloading and Preparing Datasets
Run the following scripts to download the object files, then render them into images.
Note, you will need an account at [shapenet.org](https://shapenet.org) to download the 
shapenet object files.
```
cd datasets
python render_objects.py --dataset=modelnet40 --mode=depth --num_views=60
python render_objects.py --dataset=modelnet40 --mode=gray --num_views=60
python render_objects.py --dataset=shapenet55 --mode=rgb --num_views=60
cd ..
```
Rendering can take several hours for all object classes, you can selectively render
object classes using `--objects` arg.

## Training Orientation Prediction
```
python run.py --task=orientation --dataset=modelnet40 --objects=select --mode=depth
```

## Training Shape Classification
```
python run.py --task=classification --dataset=modelnet40 --objects=all --mode=depth
```
