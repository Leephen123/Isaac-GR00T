import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

# def quat_rotate_inverse_on_gravity(q: np.ndarray) -> np.ndarray:
#     qw, qx, qy, qz = q
#     gravity_orientation = np.zeros(3, dtype=np.float32)
#     gravity_orientation[0] = 2 * (qx*qz + qw*qy)
#     gravity_orientation[1] = 2 * (qy*qz - qw*qx)
#     gravity_orientation[2] = qw*qw - qx*qx - qy*qy + qz*qz
#     return gravity_orientation

def quat_rotate_inverse_on_gravity(q: np.ndarray) -> np.ndarray:
    v = np.array([0., 0., -1.], dtype=np.float32)
    
    q_w = q[0]
    q_vec = q[1:]
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.dot(q_vec, v)
    return a - b + c

def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def quat_normalize(q):
    x = q
    z = (x[..., 3:] < 0).astype(np.float32)
    x = x * (1 - z * 2)
    norm = np.linalg.norm(x, axis=-1)
    return x / norm[..., np.newaxis].clip(min=1e-8)


def get_heading_from_quat(quat: np.ndarray) -> float:
    vec = np.array([1.0, 0.0, 0.0])

    w = quat[0]
    xyz = quat[1:]
    t = np.cross(xyz, vec) * 2
    forward = vec + w * t + np.cross(xyz, t)

    heading = np.arctan2(forward[1], forward[0])
    return heading

