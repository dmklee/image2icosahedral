import argparse
from typing import Optional, List, Callable
import os
import subprocess
import glob
import logging
import concurrent.futures
from tqdm import tqdm
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import pyrender
from io import BytesIO

logging.getLogger('pyembree').disabled = True
logging.getLogger('trimesh').disabled = True


SYMMETRIC_CATEGORIES = {'bottle', 'cup', 'stool', 'cone', 'bowl', 'vase'}


class ToMesh:
    def __call__(self, path):
        mesh = trimesh.load_mesh(path)
        if isinstance(mesh, trimesh.Scene):
            dump = mesh.dump()
            mesh = dump.sum()

        else:
            mesh.remove_degenerate_faces()
            mesh.fix_normals()
            mesh.fill_holes()
            mesh.remove_duplicate_faces()
            mesh.remove_infinite_values()
            mesh.remove_unreferenced_vertices()
            mesh.apply_translation(-mesh.centroid)

            r = np.percentile(np.linalg.norm(mesh.vertices, axis=-1), 98)
            mesh.apply_scale(1 / r)

        return mesh


class RenderDepth:
    def __init__(self, img_size, distance=3.2, fov=45, to_int=True):
        self.img_size = img_size
        self.to_int = to_int
        self.distance = distance
        self.fov = fov

    def __call__(self, mesh, cam_rotmat):
        scene = mesh.scene()

        scene.camera.resolution = (self.img_size, self.img_size)
        scene.camera.fov = self.fov * scene.camera.resolution/scene.camera.resolution.max()
        tfm = np.eye(4)
        tfm[:3,:3] = cam_rotmat
        tfm[:3,3] = tfm[:3,:3].dot(np.array((0, 0, self.distance)))

        scene.camera_transform = tfm

        origins, vectors, pixels = scene.camera_rays()

        points, index_ray, index_tri = mesh.ray.intersects_location(origins, vectors,
                                                                    multiple_hits=False)

        depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])

        pixel_ray = pixels[index_ray]

        if self.to_int:
            a = np.full(scene.camera.resolution, 255, dtype=np.uint8)
            depth_norm = 1 - (depth - self.distance + 1)/2
            depth_norm = np.clip((255 * depth_norm).round(), 0, 255).astype(np.uint8)
            a[pixel_ray[:,0], pixel_ray[:,1]] = depth_norm
        else:
            a = np.zeros(scene.camera.resolution, dtype=np.float32)
            a[pixel_ray[:,0], pixel_ray[:,1]] = depth

        return a[None]


class RenderRGB():
    def __init__(self, img_size, distance=3.2, fov=45, to_int=False):
        self.img_size = img_size
        self.to_int = to_int
        self.distance = distance
        self.fov = fov

    def __call__(self, mesh, cam_rotmat):

        tfm = np.eye(4)
        tfm[:3,:3] = cam_rotmat
        tfm[:3,3] = tfm[:3,:3].dot(np.array((0, 0, self.distance)))

        light_tfm = np.eye(4)
        light_tfm[:3,:3] = cam_rotmat
        light_tfm[:3,3] = light_tfm[:3,:3].dot(np.array((0, 0, self.distance)))

        scene = pyrender.Scene(bg_color=[1., 1., 1.])
        material = pyrender.material.MetallicRoughnessMaterial(
                                alphaMode='BLEND',
                                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                                metallicFactor=0.1,
                                roughnessFactor=0.2
                            )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        light = pyrender.PointLight(color=[1., 1., 1.], intensity=8.0)
        cam = pyrender.PerspectiveCamera(yfov=np.radians(self.fov))
        nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nl = pyrender.Node(light=light, matrix=light_tfm)
        nc = pyrender.Node(camera=cam, matrix=tfm)
        scene.add_node(nm)
        scene.add_node(nl)
        scene.add_node(nc)

        r = pyrender.OffscreenRenderer(viewport_width=self.img_size,
                                       viewport_height=self.img_size,
                                       point_size=1.0)

        color, depth = r.render(scene)
        r.delete()
        return np.transpose(color, (2,0,1))


def _render(mesh_file: str,
            output_mode: str,
            num_views: int,
            img_size: int=0,
           ):
    mesh = ToMesh()(mesh_file)
    render_fn = {'rgb' : RenderRGB(img_size),
                 'gray' : lambda m,r: np.mean(RenderRGB(img_size)(m,r), axis=0, keepdims=True).astype(np.uint8),
                 'depth' : RenderDepth(img_size),
                }[output_mode]

    view_mtxs = R.random(num_views).as_matrix()
    imgs = np.array([render_fn(mesh, vm) for vm in view_mtxs])
    return imgs, view_mtxs


def render_objects(dest: str,
                   glob_query: str,
                   output_mode: str,
                   img_size: int=128,
                   num_views: int=60,
                   strip_z_axis_rot: bool=False,
                   seed: int=0,
                   num_procs: int=4,
                  ):
    assert output_mode in ('rgb', 'gray', 'depth')

    np.random.seed(seed)
    mesh_files = glob.glob(glob_query)

    if len(mesh_files) == 0:
        raise FileNotFoundError(f'mesh files could not be found: {glob_query}')

    data = dict(output_mode=output_mode,
                img_size=img_size,
                num_views=num_views)

    pbar = tqdm(total=len(mesh_files))
    count = 0

    all_imgs = []
    all_view_mtxs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as exe:
        futures = [exe.submit(_render, f, **data) for f in mesh_files]
        for future in concurrent.futures.as_completed(futures):
            imgs, view_mtxs = future.result()
            all_imgs.extend(imgs)

            if strip_z_axis_rot:
                euler = R.from_matrix(view_mtxs).as_euler('xyz')
                rot_z = R.from_euler('z', euler[:,2]).as_matrix()
                view_mtxs = np.matmul(np.transpose(rot_z, (0,2,1)), np.array(view_mtxs))

            all_view_mtxs.extend(view_mtxs)
            pbar.update(1)

    data['img'] = torch.from_numpy(np.array(all_imgs))
    data['view_mtx'] = torch.from_numpy(np.array(all_view_mtxs)).float()
    pbar.close()

    torch.save(data, dest)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['modelnet40', 'shapenet55'])
    parser.add_argument('--img_mode', type=str, required=True,
                        choices=['depth', 'gray', 'rgb'])
    parser.add_argument('--num_views', type=int, default=60,
                        help='number of renderings per object model')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--objects', nargs='+', default=None,
                        help='only render some objects from the dataset')
    parser.add_argument('--num_procs', type=int, default=4,
                        help='number of parallel processes to use for rendering')
    args = parser.parse_args()

    # download if needed
    if args.dataset == 'modelnet40':
        download_link = 'https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar'

    elif args.dataset == 'shapenet55':
        download_link = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip'

    data_folder = args.dataset
    if not os.path.exists(data_folder):
        print('Downloading and extracting object models...')
        subprocess.run(['wget', download_link])
        zipped_fname = os.path.split(download_link)[-1]
        os.mkdir(data_folder)
        if download_link[-3:] == 'zip':
            subprocess.run(['unzip', '-d', data_folder, zipped_fname])
        else:
            subprocess.run(['tar', '-xf', zipped_fname, '--directory', data_folder])

        subprocess.run(['rm', zipped_fname])

    # render each object category
    if args.objects is None:
        args.objects = os.listdir(data_folder)

    for obj_name in args.objects:
        if os.path.isdir(os.path.join(data_folder, obj_name)):
            for mode in ('train', 'test'):
                print(f'Rendering {obj_name}-{mode}...')
                dest = os.path.join(data_folder,
                                    f"{mode}_{obj_name.replace('_', '-')}_{args.img_mode}.pt")
                render_objects(dest=dest,
                               glob_query=os.path.join(data_folder, obj_name, mode, '*.off'),
                               output_mode=args.img_mode,
                               img_size=args.img_size,
                               num_views=args.num_views,
                               strip_z_axis_rot=obj_name in SYMMETRIC_CATEGORIES,
                               seed=args.seed,
                               num_procs=args.num_procs)
