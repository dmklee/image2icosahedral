import argparse
from typing import Optional, List, Callable
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torchvision

from src.icosahedron import Icosahedron

MODELNET40_CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
                      'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
                      'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard',
                      'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
                      'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                      'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe',
                      'xbox']
MODELNET40_CLASSES_SELECT = ['desk', 'bottle', 'sofa', 'toilet', 'car',
                             'chair', 'stool', 'airplane', 'guitar', 'bench']
SHAPENET55_CLASSES_SELECT = ['guitar', 'bed', 'bottle', 'bowl', 'clock', 'chair',
                             'file_cabinet', 'airplane']


class RandomDepthShift:
    def __init__(self, mu=0., sigma=0.05):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, img):
        return img + np.random.normal(loc=self.mu, scale=self.sigma)


class RenderedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_name,
                 task,
                 objects: List,
                 mode='train',
                 img_mode='depth',
                 shift_aug=0,
                 depth_aug=0,
                ):
        self.task = task

        if 'select' in objects:
            objects =  MODELNET40_CLASSES_SELECT if dataset_name=='modelnet40' else SHAPENET55_CLASSES_SELECT

        if 'all' in objects:
            objects =  MODELNET40_CLASSES if dataset_name=='modelnet40' else SHAPENET55_CLASSES


        paths = [f'datasets/{dataset_name}/{mode}_{ob}_{img_mode}.pt' for ob in objects]

        imgs = []
        view_mtxs = []
        classes = []
        for i, path in enumerate(paths):
            try:
                new_data = torch.load(path)
            except:
                print(f'ERROR: could not find file `{path}`.')
                exit()

            N = len(new_data['img'])
            imgs.append(new_data['img'])
            view_mtxs.append(new_data['view_mtx'])
            classes.append(torch.full((N,), i, dtype=torch.long))

        self.data = {'img' : torch.cat(imgs),
                     'view_mtx' : torch.cat(view_mtxs),
                     'class' : torch.cat(classes)}

        self.img_transforms = None
        if mode == 'train':
            self.img_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomAffine(degrees=[0,0],
                                                    translate=2*[shift_aug]),
                RandomDepthShift(sigma=depth_aug),
            ])

        self.img_shape = self.data['img'][0].shape
        self.num_classes = len(objects)
        self.class_names = objects
        self.ico_group_rotmats = Icosahedron().rotation_matrices.copy()

    def __getitem__(self, index):
        img = self.data['img'][index].to(torch.float32) / 255.

        obj_rotmat = self.data['view_mtx'][index].numpy()
        obj_rotmat = obj_rotmat.T

        if self.img_transforms is not None:
            img = self.img_transforms(img)

        class_index = self.data['class'][index]

        # calculate nearest group and rotation offset
        trace = np.trace(np.matmul(self.ico_group_rotmats, obj_rotmat.T),
                         axis1=1, axis2=2)
        nearest_group_id = np.argmax(trace)
        rot_offset = obj_rotmat.dot(self.ico_group_rotmats[nearest_group_id].T)

        group_index = torch.tensor(nearest_group_id, dtype=torch.long)

        rot_offset = torch.tensor(rot_offset, dtype=torch.float32)

        if self.task == 'classification':
            return img, class_index

        return img, class_index, group_index, rot_offset

    def __len__(self):
        return len(self.data['img'])



