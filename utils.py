import os
import math
from math import cos, sin

import numpy as np
import torch
#from torch.utils.serialization import load_lua
import scipy.io as sio
import cv2
from scipy.spatial.transform import Rotation

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
        
    return out
        
#in a batch*5, axis int
def stereographic_unproject(a, axis=None):
    """
	Inverse of stereographic projection: increases dimension by one.
	"""
    batch=a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a,2).sum(1) #batch
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1]+1).cuda()) #batch*6
    unproj = 2*a/(s2+1).view(batch,1).repeat(1,a.shape[1]) #batch*5
    if(axis>0):
        ans[:,:axis] = unproj[:,:axis] #batch*(axis-0)
    ans[:,axis] = (s2-1)/(s2+1) #batch
    ans[:,axis+1:] = unproj[:,axis:]	 #batch*(5-axis)		# Note that this is a no-op if the default option (last axis) is used
    return ans

#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

# a batch*5
# out batch*3*3
def compute_rotation_matrix_from_ortho5d(a):
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = torch.autograd.Variable(torch.FloatTensor(proj_scale_np).cuda()).view(1, 3).repeat(batch, 1)  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = compute_rotation_matrix_from_ortho6d(b)
    return matrix


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally

    this code part from https://github.com/hassony2/manopth/blob/master/manopth/rot6d.py
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices, full_range=False):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()

    '''2023.01.15'''
    for i in range(len(sy)):  # expand y (yaw angle) range into (-180, 180)
        if R[i, 0, 0] < 0 and full_range:
            sy[i] = -sy[i]
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
    # print('out_euler', out_euler)
        
    return out_euler

# def compute_euler_angles_from_rotation_matrices(rotation_matrices):
#     batch = rotation_matrices.shape[0]
#     R = rotation_matrices.detach().cpu().numpy()  # Convert to NumPy array
#
#     rotations = Rotation.from_matrix(R)
#     euler_angles = rotations.as_euler('xyz', degrees=False)
#
#     out_euler = torch.tensor(euler_angles, device=rotation_matrices.device)
#     out_euler = out_euler.view(batch, 3)
#
#     return out_euler

def compute_euler_angles_from_quaternion(quaternions): #REDHWAN
    batch = quaternions.shape[0]
    # q = quaternions.detach().cpu().numpy() # Convert to NumPy array
    q = quaternions.detach().cpu().numpy()

    rotations = Rotation.from_quat(q)
    euler_angles = rotations.as_euler('xyz',  degrees=False)

    # out_euler = torch.tensor(euler_angles, device=quaternions.device)
    out_euler = torch.tensor(euler_angles.copy(), device=quaternions.device)
    out_euler = out_euler.view(batch, 3)

    return out_euler




# def compute_euler_angles_from_quaternion(quaternions, sequence='xyz'):
#     """
#     Convert a batch of quaternions to Euler angles.
#
#     Args:
#         quaternions: Tensor of shape (batch_size, 4). Batch of quaternions.
#         sequence: String specifying the rotation sequence. Default is 'xyz'.
#
#     Returns:
#         euler_angles: Tensor of shape (batch_size, 3). Batch of Euler angles.
#     """
#     batch_size = quaternions.shape[0]
#     print("Batch size:", batch_size)
#     # if batch_size ==4:
#     #     quaternions = quaternions[0]
#     q = quaternions.detach().cpu().numpy()  # Convert to NumPy array
#
#     # rotations = Rotation.from_quat(quaternions)
#     rotations = Rotation.from_quat(q)
#     print("q shape & type :", q.shape, type(q))
#     print("q :", q)
#     euler_angles = rotations.as_euler(sequence, degrees=False)
#     print("Shape of euler_angles:", euler_angles.shape, euler_angles)
#     euler_angles = torch.tensor(euler_angles, device=quaternions.device)
#     euler_angles = euler_angles.view(batch_size, 3)
#
#
#     return euler_angles

# def compute_euler_angles_from_quaternion(quaternions, sequence='xyz'):
#     batch_size = quaternions.shape[0]
#     print("Batch size:", batch_size)
#
#     q = quaternions.detach().cpu().numpy()  # Convert to NumPy array
#     rotations = Rotation.from_quat(q)
#
#     euler_angles = torch.tensor(rotations.as_euler(sequence, degrees=False), device=quaternions.device)
#     euler_angles = euler_angles.view(batch_size, 3)
#
#     return euler_angles

# def compute_euler_angles_from_quaternion(quaternions, sequence='xyz'):
#     batch_size = quaternions.shape[0]
#     print("Batch size:", batch_size)
#
#     q = quaternions.detach().cpu().numpy()  # Convert to NumPy array
#     rotations = Rotation.from_quat(q)
#
#     euler_angles = torch.tensor(rotations.as_euler(sequence, degrees=False), device=quaternions.device)
#     print('euler_angles', euler_angles.shape, euler_angles)
#
#     # If batch_size is 1, unsqueeze the tensor to make it [1, 3]
#     if batch_size == 4:
#         euler_angles = euler_angles.unsqueeze(0)
#         print('euler_angles', euler_angles.shape, euler_angles)
#         # euler_angles = euler_angles.view(1, -1)
#
#     return euler_angles

# def compute_euler_angles_from_quaternion(quaternions, sequence='xyz'):
#     batch_size = quaternions.shape[0]  # Ensure that batch_size is correctly calculated
#     print("Batch size:", batch_size)
#
#     q = quaternions.detach().cpu().numpy()  # Convert to NumPy array
#
#     rotations = Rotation.from_quat(q)
#     euler_angles = rotations.as_euler(sequence, degrees=False)
#     print("Shape of euler_angles:", euler_angles.shape)
#
#     if batch_size == 1:
#         euler_angles = euler_angles.view(euler_angles.size(0), 3)  # Reshape single quaternion to (1, 3)
#
#     euler_angles = torch.tensor(euler_angles, device=quaternions.device)
#     #
#     # assert batch_size == euler_angles.shape[0], "Mismatch in batch_size and euler_angles shape"
#     # euler_angles = euler_angles.view(batch_size, 3)  # Reshape to [batch_size, 3]
#
#     return euler_angles


# def normalize_quaternion(q):
#     return q / torch.norm(q)

# def quaternion_to_euler_xyz(q):
#     q = normalize_quaternion(q)
#     q0, q1, q2, q3 = q
#
#     roll = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
#     pitch = torch.asin(2 * (q0 * q2 - q3 * q1))
#     yaw = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
#
#     return roll, pitch, yaw

# def compute_euler_angles_from_quaternion(q):
#     # Normalize the quaternion
#     q_norm = q / torch.norm(q, dim=1, keepdim=True)
#
#     # Extract quaternion components
#     q0 = q_norm[:, 0]
#     q1 = q_norm[:, 1]
#     q2 = q_norm[:, 2]
#     q3 = q_norm[:, 3]
#
#     # Compute rotation matrix elements
#     R00 = 1 - 2 * (q2 ** 2 + q3 ** 2)
#     R01 = 2 * (q1 * q2 - q0 * q3)
#     R02 = 2 * (q1 * q3 + q0 * q2)
#     R10 = 2 * (q1 * q2 + q0 * q3)
#     R11 = 1 - 2 * (q1 ** 2 + q3 ** 2)
#     R12 = 2 * (q2 * q3 - q0 * q1)
#     R20 = 2 * (q1 * q3 - q0 * q2)
#     R21 = 2 * (q2 * q3 + q0 * q1)
#     R22 = 1 - 2 * (q1 ** 2 + q2 ** 2)
#
#     # Compute the Euler angles
#     sy = torch.sqrt(R00 * R00 + R10 * R10)
#     singular = sy < 1e-6
#     singular = singular.float()
#
#     x = torch.atan2(R21, R22)
#     y = torch.atan2(-R20, sy)
#     z = torch.atan2(R10, R00)
#
#     xs = torch.atan2(-R12, R11)
#     ys = torch.atan2(-R20, sy)
#     zs = torch.zeros_like(x)
#
#     # Combine the Euler angles based on singularities
#     euler_angles = torch.zeros(q.shape[0], 3, dtype=q.dtype, device=q.device)
#     euler_angles[:, 0] = x * (1 - singular) + xs * singular
#     euler_angles[:, 1] = y * (1 - singular) + ys * singular
#     euler_angles[:, 2] = z * (1 - singular) + zs * singular
#
#     return euler_angles

# def compute_euler_angles_from_quaternion(quaternions, sequence='xyz'):
#     q = quaternions.detach().cpu().numpy()  # Convert to NumPy array
#     rotations = Rotation.from_quat(q)
#
#     euler_angles = rotations.as_euler(sequence, degrees=False)
#
#
#     return euler_angles

# def compute_euler_angles_from_quaternion(quaternions, full_range=False, use_gpu=True, gpu_id=0):
#     batch = quaternions.shape[0]
#     q = quaternions
#
#     # Extract quaternion components
#     w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
#
#     # Convert quaternion to Euler angles
#     ys = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
#     xs = torch.asin(2 * (w * y - z * x))
#     zs = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
#
#     # Handle the case of full range yaw angle (-180, 180)
#     if full_range:
#         ys = ys * 180 / np.pi
#         ys = (ys + 180) % 360 - 180
#
#     if use_gpu:
#         out_euler = torch.autograd.Variable(torch.zeros(batch, 3).cuda(gpu_id))
#     else:
#         out_euler = torch.autograd.Variable(torch.zeros(batch, 3))
#
#     out_euler[:, 0] = xs
#     out_euler[:, 1] = ys
#     out_euler[:, 2] = zs
#
#     return out_euler


def convert_quaternions_to_rotation_matrices(quaternions):
    batch = quaternions.shape[0]
    # print("Batch:", batch)
    q = quaternions.detach().cpu().numpy()  # Convert to NumPy array

    rotations = Rotation.from_quat(q)
    rotation_matrices = rotations.as_matrix()

    out_matrices = torch.tensor(rotation_matrices, device=quaternions.device)
    # out_matrices = out_matrices.view(batch, 3, 3)
    # print('out_matrices', out_matrices.shape, out_matrices)
    if out_matrices.numel() == 9:
        out_matrices = out_matrices.view(1, 3, 3)  # Assuming a single rotation matrix
    else:
        out_matrices = out_matrices.view(batch, 3, 3)

    return out_matrices





def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R

def get_q(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    """
       Get rotation matrix from three rotation angles (radians). right-handed.

       Args:
           x: Tensor of shape (batch_size,). X-axis rotation angles (in radians).
           y: Tensor of shape (batch_size,). Y-axis rotation angles (in radians).
           z: Tensor of shape (batch_size,). Z-axis rotation angles (in radians).

       Returns:
           R: Tensor of shape (batch_size, 3, 3). Rotation matrices.
       """
    # Create rotation objects from Euler angles
    r = Rotation.from_euler('xyz', np.stack([x, y, z], axis=-1), degrees=False)
    # r = Rotation.from_euler('xyz', degrees=False)

    # Convert rotation objects to rotation matrices
    q = torch.tensor(r.as_quat(), dtype=torch.float32)

    return q

#=============== it is working fine using numpy========================================
# def quaternion_multiply(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return np.array([w, x, y, z])

# def quaternion_multiply(q1, q2):
#     w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
#     w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
#
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#
#     return np.concatenate([w, x, y, z], axis=-1)
#
# def get_q(x, y, z):
#     ''' Get quaternion from three rotation angles (radians). right-handed.
#     Args:
#         x, y, z: Rotation angles around x, y, z axes (in radians).
#     Returns:
#         q: Quaternion [w, x, y, z] representing the rotation.
#     '''
#     # Convert the individual rotation angles to quaternions
#     qx = np.array([np.cos(x / 2), np.sin(x / 2), 0, 0])  # Rotation around x-axis
#     qy = np.array([np.cos(y / 2), 0, np.sin(y / 2), 0])  # Rotation around y-axis
#     qz = np.array([np.cos(z / 2), 0, 0, np.sin(z / 2)])  # Rotation around z-axis
#
#     # Combine the quaternions using quaternion multiplication
#     q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
#     q = quaternion_multiply(q, qx)
#     q = quaternion_multiply(q, qy)
#     q = quaternion_multiply(q, qz)
#
#     return q

#===============================================================================================
#=============== it is working fine using torch========================================
# def quaternion_multiply(q1, q2):
#     w1, x1, y1, z1 = q1
#     w2, x2, y2, z2 = q2
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#     z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#     return torch.tensor([w, x, y, z])
# def get_q(x, y, z):
#     ''' Get quaternion from three rotation angles (radians). right-handed.
#     Args:
#         x, y, z: Rotation angles around x, y, z axes (in radians).
#     Returns:
#         q: Quaternion [w, x, y, z] representing the rotation.
#     '''
#     # Convert the individual rotation angles to tensors
#     x = torch.tensor(x)
#     y = torch.tensor(y)
#     z = torch.tensor(z)
#
#     # Convert the tensors to quaternions
#     qx = torch.tensor([torch.cos(x / 2), torch.sin(x / 2), 0, 0])  # Rotation around x-axis
#     qy = torch.tensor([torch.cos(y / 2), 0, torch.sin(y / 2), 0])  # Rotation around y-axis
#     qz = torch.tensor([torch.cos(z / 2), 0, 0, torch.sin(z / 2)])  # Rotation around z-axis
#
#     # Combine the quaternions using quaternion multiplication
#     q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
#     q = quaternion_multiply(q, qx)
#     q = quaternion_multiply(q, qy)
#     q = quaternion_multiply(q, qz)
#
#     return q
# def get_q(x,y,z):
#
#     # Create rotation objects from Euler angles
#     r =  Rotation.from_euler('xyz', [x, y, z], degrees=False)
#     # r = Rotation.from_euler('xyz', degrees=False)
#
#     # Convert rotation objects to rotation matrices
#     R = r.as_quat()
#
#     return R

# def compute_euler_angles_from_quaternion(quaternions): #REDHWAN
#     w, x, y, z = torch.split(quaternions, 1, dim=-1)
#
#     x2 = x * x
#     y2 = y * y
#     z2 = z * z
#     w2 = w * w
#
#     t0 = 2.0 * (w * x + y * z)
#     t1 = 1.0 - 2.0 * (x2 + y2)
#     t2 = 2.0 * (w * y - z * x)
#     t2 = torch.clamp(t2, -1.0, 1.0)  # Clamp t2 to the valid range (-1.0, 1.0)
#     t3 = 2.0 * (w * z + x * y)
#     t4 = 1.0 - 2.0 * (y2 + z2)
#
#     roll = torch.atan2(t3, t4)
#     pitch = torch.asin(t2)
#     yaw = torch.atan2(t0, t1)
#
#     # Convert angles to degrees if desired
#     # roll = torch.degrees(roll)
#     # pitch = torch.degrees(pitch)
#     # yaw = torch.degrees(yaw)
#
#     # Return Euler angles as torch tensor
#     return torch.cat([roll, pitch, yaw], dim=-1)


# quaternion batch*4 #https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/Inverse_Kinematics/code/tools.py#LL413C1-L413C68
def compute_rotation_matrix_from_quaternion(quaternion): ###Redhwan added it
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


# def convert_quaternion_euler(quaternions):
#     # Convert quaternions to Euler angles
#     rotations = Rotation.from_quat(quaternions)
#     euler_angles = rotations.as_euler('xyz', degrees=False)
#
#     # Convert Euler angles to quaternions
#     # rotations = Rotation.from_euler('xyz', euler_angles, degrees=True)
#     # converted_quaternions = rotations.as_quat()
#     # return euler_angles, converted_quaternions
#     return euler_angles

# from scipy.spatial.transform import Rotation

# import numpy as np
# from scipy.spatial.transform import Rotation

# def convert_quaternion_euler(quaternions, to_quaternion=False):
#     if to_quaternion:
#         # Convert Euler angles to quaternions
#         rotations = Rotation.from_euler('xyz', quaternions.cpu().numpy(), degrees=False)
#         converted_quaternions = rotations.as_quat()
#         return converted_quaternions
#     else:
#         # Convert quaternions to Euler angles
#         rotations = Rotation.from_quat(quaternions.cpu().numpy())
#         euler_angles = rotations.as_euler('xyz', degrees=False)
#         return euler_angles

def quaternion_to_euler(quaternion):
    # Normalize quaternion
    quaternion /= torch.norm(quaternion, dim=1, keepdim=True)

    # Extract individual components
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Compute Euler angles
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * np.pi / 2, torch.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Convert radians to degrees
    roll, pitch, yaw = torch.rad2deg(roll), torch.rad2deg(pitch), torch.rad2deg(yaw)

    # return torch.stack((roll, pitch, yaw), dim=1)
    return torch.stack(( pitch, yaw, roll), dim=1)


# Function to handle both single image and batch of images
def convert_quaternion_to_euler(quaternions):
    if len(quaternions.shape) == 1:  # Single quaternion
        return quaternion_to_euler(quaternions.unsqueeze(0))
    elif len(quaternions.shape) == 2:  # Batch of quaternions
        return quaternion_to_euler(quaternions)

