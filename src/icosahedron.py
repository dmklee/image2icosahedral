from typing import Tuple
import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R

GR = ( 1 + np.sqrt(5) ) / 2.
N_VERTICES = 12
N_EDGES = 30
N_FACES = 20

NUM_VERTICES_BY_LEVEL = {0: 12, 1: 42, 2: 162, 3: 642}


def project_new_plane(points, z_axis, init_axis=[1,0,0]):
    y_axis = np.cross([1,0,0], z_axis)
    x_axis = np.cross(z_axis, y_axis)
    axis = np.stack([x_axis, y_axis, z_axis])
    points_in_plane = np.dot(points, axis.T)
    return points


class Icosahedron:
    def __init__(self, level: int=0):
        self.edge_length = 2 / np.sqrt(GR * np.sqrt(5)) # this way radius=1

        ico_vertices = []
        dod_vertices = [] # faces of ico
        for a,b in itertools.product([-1,1],[-1,1]):
            ico_vertices.extend(((0, a, b*GR), (a, b*GR, 0), (a*GR, 0, b)))
            dod_vertices.extend(((0, a/GR, b*GR), (a/GR, b*GR, 0),(a*GR, 0, b/GR)))
        dod_vertices.extend(itertools.product([-1,1],[-1,1],[-1,1]))

        # normalize so they have radius = 1
        self.vertices = np.array(ico_vertices) * self.edge_length/2
        self.dod_vertices = np.array(dod_vertices) / np.sqrt(3)

        self.edge_indices = self.find_edge_indices(self.vertices, self.edge_length)

        self.all_vertices, self.all_edge_indices = self.split_edges(self.vertices,
                                                                    self.edge_indices,
                                                                    self.edge_length,
                                                                    level)
        self.minor_edge_length = self.edge_length / (2**level)

        self.rotation_groups = np.array((
            # rotation by 2pi/5 around (0,1,phi)
            (( (np.sqrt(5)-1)/4, -(np.sqrt(5)+1)/4,              1/2),
             ( (np.sqrt(5)+1)/4,               1/2, (np.sqrt(5)-1)/4),
             (             -1/2,  (np.sqrt(5)-1)/4, (np.sqrt(5)+1)/4)),
            #rotation by 2pi/3 around (1,1,1)
            ((0, 0, 1),
             (1, 0, 0),
             (0, 1, 0)),
            # rotation by pi around (0,0,1)
            ((1, 0, 0),
             (0,-1, 0),
             (0, 0,-1)),
        ))
        self.rotation_group_orders = (5,3,2)

        self.generate_group()

        phi = np.arctan2(*self.all_vertices[0,1:])
        rot = np.array(((1,           0,            0),
                        (0, np.cos(phi), -np.sin(phi)),
                        (0, np.sin(phi),  np.cos(phi))))
        tfm_vertices = np.dot(rot, self.all_vertices.T).T
        mask = tfm_vertices[:,2] > 1e-6
        self.halfmesh_locs = tfm_vertices[mask][:,:2]
        self.halfmesh_mask = mask

        self.local_mask = R.from_matrix(self.rotation_matrices).magnitude() <= 2*np.pi/5

        self.num_vertices = NUM_VERTICES_BY_LEVEL[level]

    def generate_group(self):
        #ToDo: clean this up later
        rotation_matrices = [np.eye(3)]
        for z in range(5):
            for i,j,k in itertools.product(*[range(i) for i in self.rotation_group_orders]):
                for a,b,c in itertools.permutations([0,1,2], 3):
                    rotA = np.linalg.matrix_power(self.rotation_groups[a], i)
                    rotB = np.linalg.matrix_power(self.rotation_groups[b], j)
                    rotC = np.linalg.matrix_power(self.rotation_groups[c], k)
                    rot = np.dot(rotA, rotB.dot(rotC))
                    rot = rot.dot(np.linalg.matrix_power(self.rotation_groups[0], z))

                    # check it is unique
                    if (np.linalg.norm(np.subtract(rotation_matrices, rot), axis=(1,2)) > 1e-6).all():
                        rotation_matrices.append(rot)

        self.rotation_matrices = np.array(rotation_matrices)

        rotated_vertices = np.swapaxes(self.rotation_matrices.dot(self.all_vertices.T), 1,2)
        distances = np.linalg.norm(self.all_vertices[None,:,None]-rotated_vertices[:,None], axis=-1)
        self.quotient_reps = (distances < 1e-6).astype(np.float32)

        rotated_groups = np.moveaxis(np.dot(self.rotation_matrices, np.moveaxis(self.rotation_matrices, 0, -1)), -1, 1)
        distances = np.linalg.norm(self.rotation_matrices[:,None,None] - rotated_groups[None], axis=(-1,-2))
        self.regular_reps = (distances < 1e-6).astype(np.float32)

    def split_edges(self,
                    vertices: np.ndarray,
                    edge_indices: np.ndarray,
                    edge_length: float,
                    level: int,
                   ) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(level):
            new_vertices = []
            new_edge_indices = []
            edge_length = edge_length/2
            for i,j in edge_indices:
                midpt = np.mean([vertices[i],vertices[j]], axis=0)
                new_vertices.append(midpt)
            vertices = np.concatenate([vertices, np.array(new_vertices)], axis=0)
            edge_indices = self.find_edge_indices(vertices, edge_length)

        return vertices, edge_indices

    def find_edge_indices(self, vertices: np.ndarray, edge_length: float) -> np.ndarray:
        '''pairs up any edges that are within edge length in cartesian space'''
        edge_indices = []
        for i in range(len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices, axis=1)
            neighbors = np.where(np.isclose(dist, edge_length))[0]
            for vi in neighbors:
                if i < vi:
                    edge_indices.append((i, vi))
        return np.array(edge_indices, dtype=int)

