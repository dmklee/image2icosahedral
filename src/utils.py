import numpy as np
import torch
import torch.nn.functional as F


def fps(point: np.ndarray, npoint: int):
    """
    Input:
    xyz: pointcloud data, [N, D]
    npoint: number of samples
    Return:
    centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    indices = centroids.astype(np.int32)
    point = point[indices]
    return point, indices

def rotation_error(rotA, rotB, mode='l2'):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns error tensor of shape (*)
    '''
    assert mode in ('l2', 'trace', 'neg-trace', 'angle')
    if mode == 'l2':
        return torch.linalg.norm(rotA - rotB, dim=(-1,-2))

    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    if mode == 'trace':
        return trace

    if mode == 'neg-trace':
        return -1 * trace

    if mode == 'angle':
        return torch.arccos(torch.clamp( (trace - 1)/2, -1, 1))

def exp_so3(w, theta, taylor_approx=True):
    '''
    performs exponential map from lie algebra of SO(3) to element of SO(3)
    https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    '''
    theta = np.pi / 5 * torch.sigmoid(theta)
    w = theta * F.normalize(w, dim=2)

    if taylor_approx:
        A = 1 - theta.pow(2) / 6
        B = 0.5 - theta.pow(2) / 24
    else:
        A = torch.sin(theta) / (theta + 1e-8)
        B = (1 - torch.cos(theta)) / (theta.pow(2) + 1e-8)

    x, y, z = torch.split(w, [1,1,1], dim=2)
    x2 = x*x
    y2 = y*y
    z2 = z*z
    xy = x*y
    yz = y*z
    xz = x*z

    row0 = torch.cat((1 - B*(z2+y2),   -A*z + B*xy,    A*y + B*xz), 2)
    row1 = torch.cat((   A*z + B*xy, 1 - B*(z2+x2),   -A*x + B*yz), 2)
    row2 = torch.cat((  -A*y + B*xz,    A*x + B*yz, 1 - B*(y2+x2)), 2)

    matrix = torch.cat((row0.unsqueeze(2), row1.unsqueeze(2), row2.unsqueeze(2)),
                       dim=2) #B*3*3

    return matrix

def log_so3(R, taylor_approx=True):
    tr = torch.matmul(R, R.transpose(dim0=-1, dim1=-2)).diagonal(0,-1,-2).sum(-1)
    theta = torch.arccos(torch.clamp((tr - 1)/2, -1, 1))

    if taylor_approx:
        first_term = 1/2 - theta.pow(2)/12
    else:
        first_term = theta / (2 * torch.sin(theta))

    logR = first_term * (R - R.transpose(dim0=-1, dim2=-2))
    return logR

def axisangle2rotmat(axis, angle):
    # quat: float tensor of shape (B,*,4)
    x,y,z = torch.split(axis, [1,1,1], dim=2)
    # c = torch.cos(angle)
    # s = torch.sin(angle)
    c = 1 - angle.pow(2)/2 + angle.pow(4)/24
    s = angle - angle.pow(3)/6 + angle.pow(5)/120

    row0 = torch.cat((x*x*(1-c) + c, x*y*(1-c) - z*s, x*z*(1-c) + y*s), 2)
    row1 = torch.cat((y*x*(1-c) +z*s, y*y*(1-c) + c, y*z*(1-c) - x*s), 2)
    row2 = torch.cat((z*x*(1-c) - y*s, z*y*(1-c) + x*s, z*z*(1-c) + c), 2)

    B = axis.size(0)
    matrix = torch.cat((row0.unsqueeze(2), row1.unsqueeze(2), row2.unsqueeze(2)),
                       dim=2) #B*3*3

    return matrix

def quat2rotmat(quat):
    # quat: float tensor of shape (B,*,4)
    qx, qy, qz, qw = torch.split(quat, [1,1,1,1], dim=2)

    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 2)
    row1 = torch.cat((2*xy+2*zw,   1-2*xx-2*zz,   2*yz-2*xw), 2)
    row2 = torch.cat((2*xz-2*yw,     2*yz+2*xw, 1-2*xx-2*yy), 2)

    B = quat.size(0)
    matrix = torch.cat((row0.unsqueeze(2), row1.unsqueeze(2), row2.unsqueeze(2)),
                       dim=2) #B*3*3

    return matrix

def gs_orthogonalization(vecs):
    # vecs: float tensor of shape (B,*,6)
    u, v = torch.split(vecs, [3,3], dim=2)

    u_p = u / torch.linalg.vector_norm(u, dim=2, keepdims=True)
    v_p = (v - (u_p * v).sum(dim=2, keepdims=True)*u_p)
    v_p = v_p / torch.linalg.norm(v_p, dim=2, keepdims=True)
    w_p = torch.cross(u_p, v_p, dim=2)

    return torch.stack([u_p, v_p, w_p], dim=3)

def nearest_rotmat(src, target):
    '''return index of target that is nearest to each element in src

    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    dist = rotation_error(src.unsqueeze(1),
                          target.unsqueeze(0),
                          mode='neg-trace')

    return torch.min(dist, dim=1)[1]
