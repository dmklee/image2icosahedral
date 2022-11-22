import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import EqEncoder, Encoder, FmapSampler
from src.ico_group_conv import FeatureSphereLibrary, DynamicIcoConv
from src import utils


class SO3Predictor(nn.Module):
    def __init__(self,
                 img_shape=(1, 128, 128),
                 num_classes: int=1,
                 sphere_fdim: int=128,
                 ico_level: int=1,
                 sampler_coverage: float=0.9,
                 sampler_sigma: float=0.2,
                 offset_lambda: float=10,
                 encoder_type='resnet_equiv',
                 encoder_kwargs: dict={},
                ):
        super().__init__()

        # one for probability and 6 for rotation offset via GramSchmidt
        self.output_size = 1 + 6
        self.offset_lambda = offset_lambda

        self.ico_module = DynamicIcoConv(ico_level)
        self.num_vertices = self.ico_module.num_vertices

        self.num_classes = num_classes
        self.sphere_fdim = sphere_fdim
        self.feature_spheres = FeatureSphereLibrary(num_classes,
                                                    sphere_fdim,
                                                    self.output_size,
                                                    self.num_vertices)

        if encoder_type == 'resnet':
            self.encoder = Encoder(img_shape,
                                   self.output_size * sphere_fdim,
                                   **encoder_kwargs)
        elif encoder_type == 'resnet_equiv':
            self.encoder = EqEncoder(img_shape,
                                     self.output_size * sphere_fdim,
                                     **encoder_kwargs)

        fmap_size = self.encoder.output_shape[-1]
        print('fmap size', fmap_size)

        self.projector = FmapSampler(fmap_size,
                                     ico_level=ico_level,
                                     sigma=sampler_sigma,
                                     fraction_coverage=sampler_coverage)

    def generate_filter(self, x):
        fmap = self.encoder(x)
        mesh = self.projector(fmap)
        mesh = mesh.view(mesh.size(0), mesh.size(1),
                         mesh.size(2)//self.output_size, self.output_size)

        return mesh

    def forward(self, x, o):
        mesh = self.generate_filter(x)
        mesh = torch.relu(mesh)

        sphere, bias = self.feature_spheres.get_spheres(o)

        out = self.ico_module(sphere, mesh, bias)

        act, rot = torch.split(out, [1, 6], dim=2)

        rotmat_offset = utils.gs_orthogonalization(rot)

        return act, rotmat_offset

    def generate_full_rotmat(self, group_ids, rot_offsets):
        group_rotmats = self.ico_module.rotmats[group_ids.squeeze()]
        return torch.bmm(rot_offsets, group_rotmats)

    @torch.no_grad()
    def predict(self, x, o):
        act, rotmat_offset = self.forward(x, o)
        group_ids = torch.max(act, dim=1)[1].squeeze(1)
        rotmat_offset = rotmat_offset[torch.arange(x.size(0)), group_ids]
        return self.generate_full_rotmat(group_ids, rotmat_offset)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.feature_spheres.parameters())

    def save(self, path):
        torch.save({'encoder' : self.encoder.state_dict(),
                    'spheres' : self.feature_spheres.state_dict(),
                   }, path)

    def load(self, path):
        state_dict = torch.load(path)
        # must go train mode for it to work with e2cnn models
        self.encoder.train()
        self.encoder.load_state_dict(state_dict['encoder'])
        self.feature_spheres.load_state_dict(state_dict['spheres'])

    def compute_loss(self, img, cls_idx, grp_idx, gt_rot_offset):
        act, all_offset_pred = self.forward(img, cls_idx)

        cls_loss = nn.CrossEntropyLoss()(act.squeeze(-1), grp_idx)

        grp_idx_pred = torch.max(act, dim=1)[1].squeeze(1)
        is_correct = (grp_idx_pred == grp_idx).float()
        rot_offset_pred = all_offset_pred[torch.arange(all_offset_pred.size(0)), grp_idx_pred]
        reg_loss = self.offset_lambda * torch.mean(is_correct * torch.linalg.norm(gt_rot_offset-rot_offset_pred, dim=(1,2)))

        loss = cls_loss + reg_loss

        with torch.no_grad():
            full_rotmat = self.generate_full_rotmat(grp_idx, gt_rot_offset)
            rot_offset_pred = all_offset_pred[torch.arange(all_offset_pred.size(0)), grp_idx_pred]
            full_rotmat_pred = self.generate_full_rotmat(grp_idx_pred, rot_offset_pred)

            acc = utils.rotation_error(full_rotmat, full_rotmat_pred, 'angle')

        data = dict(cls_loss=cls_loss.item(),
                    reg_loss=reg_loss.item(),
                    acc=acc.cpu().numpy())

        return loss, data


class ClassPredictor(SO3Predictor):
    def __init__(self,
                 img_shape=(1,128,128),
                 num_classes: int=1,
                 sphere_fdim: int=128,
                 ico_level: int=1,
                 sampler_coverage: float=0.9,
                 sampler_sigma: float=0.2,
                 encoder_type='resnet_equiv',
                 encoder_kwargs: dict={},
                 **kwargs,
                ):
        nn.Module.__init__(self)
        self.output_size = num_classes

        self.ico_module = DynamicIcoConv(ico_level)
        self.num_vertices = self.ico_module.num_vertices

        self.num_classes = num_classes
        self.feature_spheres = FeatureSphereLibrary(1,
                                                    sphere_fdim,
                                                    self.output_size,
                                                    self.num_vertices)

        if encoder_type == 'resnet':
            self.encoder = Encoder(img_shape,
                                   self.output_size * sphere_fdim,
                                   **encoder_kwargs)
        elif encoder_type == 'resnet_equiv':
            self.encoder = EqEncoder(img_shape,
                                     self.output_size * sphere_fdim,
                                     **encoder_kwargs)

        fmap_size = self.encoder.output_shape[-1]
        print('fmap size', fmap_size)

        self.projector = FmapSampler(fmap_size,
                                     ico_level=ico_level,
                                     sigma=sampler_sigma,
                                     fraction_coverage=sampler_coverage)

    def forward(self, x):
        mesh = self.generate_filter(x)
        mesh = torch.relu(mesh)

        sphere = self.feature_spheres.weight
        bias = self.feature_spheres.bias

        act = self.ico_module(sphere, mesh.unsqueeze(1), bias.unsqueeze(0)).squeeze(1)
        # act = (B, G, C)
        act = torch.max(act, dim=1)[0]
        return act

    @torch.no_grad()
    def predict(self, x):
        act = self.forward(x)
        return act

    def compute_loss(self, img, cls_idx):
        B = len(img)
        act = self.forward(img)

        cls_loss = nn.CrossEntropyLoss()(act, cls_idx)

        loss = cls_loss

        with torch.no_grad():
            cls_acc = (act.max(1)[1] == cls_idx).float()

        data = dict(cls_loss=cls_loss.item(),
                    reg_loss=0,
                    acc=cls_acc.cpu().numpy(),
                   )

        return loss, data
