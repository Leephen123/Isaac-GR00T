import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .log import get_logger

logger = get_logger(__name__)


def normalize_quaternion(q: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Normalize quaternions to unit length.

    Args:
        q:   (..., 4) array of quaternions (any convention)
        tol: tolerance for detecting non-unit quaternions

    Returns:
        (..., 4) normalized quaternions
    """
    norms = np.linalg.norm(q, axis=-1, keepdims=True)  # (..., 1)
    not_unit = ~np.isclose(norms, 1.0, atol=tol, rtol=0)
    if np.any(not_unit):
        bad_count = np.sum(not_unit)
        logger.warning(
            f"{bad_count} quaternion(s) have non-unit norm (tolerance={tol}). Normalizing automatically."
        )
    return q / np.clip(norms, a_min=1e-12, a_max=None)


def smooth_pose7_quat_sign(pose7_seq: np.ndarray) -> np.ndarray:
    """
    Enforce quaternion sign continuity in a pose7 sequence.

    Args:
        pose7_seq: (T, J, 7), format [x, y, z, qw, qx, qy, qz]

    Returns:
        (T, J, 7): pose7 sequence with temporally continuous quaternion signs
    """

    pose7_seq = np.asarray(pose7_seq, dtype=np.float32).copy()
    T, _, D = pose7_seq.shape

    if D != 7:
        raise ValueError(f"Expected last dim 7 ([x,y,z,qw,qx,qy,qz]), got {D}")
    if T <= 1:
        return pose7_seq

    # ensure qw >= 0 for the first frame quaternion of each joint
    quat = pose7_seq[:, :, 3:7]
    init_flip = quat[0, :, 0] < 0.0  # (J,)
    quat[:, init_flip, :] *= -1.0

    # ensure temporal sign continuity per joint.
    for t in range(1, T):
        dot = np.sum(quat[t - 1] * quat[t], axis=-1)  # (J,)
        flip = dot < 0.0
        quat[t, flip, :] *= -1.0

    pose7_seq[:, :, 3:7] = quat
    return pose7_seq


def compute_absolute(reference_pose, relative_pose):
    """
    Reconstruct absolute world-frame pose from a relative pose and a reference pose.
    This is the inverse of compute_relative.

    Args:
        relative_pose:  (N, 7), each row = [x, y, z, qw, qx, qy, qz], output of compute_relative
        reference_pose: (N, 7), each row = [x, y, z, qw, qx, qy, qz] in world frame

    Returns:
        (N, 7): absolute pose in world frame
    """
    if relative_pose.shape != reference_pose.shape:
        raise ValueError(
            f"Cannot reconstruct absolute pose: "
            f"Pose dimensions don't match ({relative_pose.shape} vs {reference_pose.shape})"
        )

    def wxyz_to_xyzw(q):
        return np.roll(q, -1, axis=-1)

    def xyzw_to_wxyz(q):
        return np.roll(q, 1, axis=-1)

    xyz_ref = reference_pose[:, :3]
    quat_ref = normalize_quaternion(reference_pose[:, 3:])
    xyz_rel = relative_pose[:, :3]
    quat_rel = normalize_quaternion(relative_pose[:, 3:])

    r_ref = Rotation.from_quat(wxyz_to_xyzw(quat_ref))
    r_rel = Rotation.from_quat(wxyz_to_xyzw(quat_rel))

    # Reconstruct yaw-only rotation of reference (same as in compute_relative)
    # Use arctan2 on the rotation matrix to avoid gimbal lock
    R_ref = r_ref.as_matrix()  # (N, 3, 3)
    yaw_ref = np.arctan2(R_ref[:, 1, 0], R_ref[:, 0, 0])  # (N,)
    r_ref_yaw = Rotation.from_euler("z", yaw_ref[:, None])

    # Inverse of: r_relative = r_ref_yaw.inv() * r_cur
    r_cur = r_ref_yaw * r_rel
    quat_abs = xyzw_to_wxyz(r_cur.as_quat())

    # Inverse of: xyz_rel = r_ref_yaw.inv().apply(xyz_cur - xyz_ref)
    xyz_abs = r_ref_yaw.apply(xyz_rel) + xyz_ref

    return np.concatenate([xyz_abs, quat_abs], axis=-1)


def interpolate_pose7(pose7_seq: np.ndarray, num_interp: int) -> np.ndarray:
    """
    Interpolate between adjacent frames in a pose7 sequence.

    xyz uses linear interpolation; quaternions use SLERP (scipy).

    Args:
        pose7_seq: (T, J, 7), format [x, y, z, qw, qx, qy, qz]
        num_interp: number of frames to insert between each adjacent pair

    Returns:
        (T + (T-1)*num_interp, J, 7) float32 array
    """
    pose7_seq = np.asarray(pose7_seq, dtype=np.float64)
    T, J, D = pose7_seq.shape
    if D != 7:
        raise ValueError(f"Expected last dim 7 ([x,y,z,qw,qx,qy,qz]), got {D}")

    if T <= 1 or num_interp <= 0:
        return pose7_seq.astype(np.float32)

    # time axis: original frames at integer positions
    t_orig = np.arange(T, dtype=np.float64)

    # insert num_interp points between each pair, keep last frame
    steps = np.linspace(0.0, 1.0, num_interp + 2)[:-1]  # exclude right endpoint
    t_new = np.concatenate([i + steps for i in range(T - 1)] + [np.array([float(T - 1)])])

    # xyz: vectorized linear interpolation
    xyz = pose7_seq[:, :, 0:3]  # (T, J, 3)
    i0 = np.floor(t_new).astype(int).clip(0, T - 2)
    alpha = (t_new - i0)[:, None, None]  # (T_new, 1, 1)
    xyz_new = (1.0 - alpha) * xyz[i0] + alpha * xyz[i0 + 1]  # (T_new, J, 3)

    # quaternion: SLERP per joint
    # data format: [qw, qx, qy, qz]; scipy Rotation expects [qx, qy, qz, qw]
    quat_wxyz = pose7_seq[:, :, 3:7]  # (T, J, 4)
    quat_xyzw = np.roll(quat_wxyz, shift=-1, axis=-1)  # (T, J, 4)

    quat_new_xyzw = np.empty((len(t_new), J, 4), dtype=np.float64)
    for j in range(J):
        slerp = Slerp(t_orig, Rotation.from_quat(quat_xyzw[:, j, :]))
        quat_new_xyzw[:, j, :] = slerp(t_new).as_quat()

    quat_new_wxyz = np.roll(quat_new_xyzw, shift=1, axis=-1)  # back to [w, x, y, z]
    result = np.concatenate([xyz_new, quat_new_wxyz], axis=-1)  # (T_new, J, 7)
    return result.astype(np.float32)


def compute_relative(reference_pose, current_pose):
    """
    Compute current_pose relative to reference_pose.

    Args:
        reference_pose: (N, 7), each row = [x, y, z, qw, qx, qy, qz] in world frame
        current_pose:   (N, 7), same format

    Returns:
        (N, 7): relative pose where
            - xyz is (xyz_current - xyz_ref) rotated into the reference yaw frame
            - quaternion is ref_yaw^{-1} * current (only yaw of reference is removed)
    """
    if reference_pose.shape != current_pose.shape:
        raise ValueError(
            f"Cannot compute relative pose: "
            f"Pose dimensions don't match ({reference_pose.shape} vs {current_pose.shape})"
        )

    # scipy's Rotation expects quaternions in xyzw order
    def wxyz_to_xyzw(q):
        return np.roll(q, -1, axis=-1)

    def xyzw_to_wxyz(q):
        return np.roll(q, 1, axis=-1)

    xyz_ref = reference_pose[:, :3]
    quat_ref = normalize_quaternion(reference_pose[:, 3:])
    xyz_cur = current_pose[:, :3]
    quat_cur = normalize_quaternion(current_pose[:, 3:])

    r_ref = Rotation.from_quat(wxyz_to_xyzw(quat_ref))
    r_cur = Rotation.from_quat(wxyz_to_xyzw(quat_cur))

    # Extract only the yaw component of reference rotation (rotation around Z axis)
    # Use arctan2 on the rotation matrix to avoid gimbal lock
    R_ref = r_ref.as_matrix()  # (N, 3, 3)
    yaw_ref = np.arctan2(R_ref[:, 1, 0], R_ref[:, 0, 0])  # (N,)
    r_ref_yaw = Rotation.from_euler("z", yaw_ref[:, None])  # yaw-only rotation

    # relative rotation: ref_yaw^{-1} * current
    r_relative = r_ref_yaw.inv() * r_cur
    quat_rel = xyzw_to_wxyz(r_relative.as_quat())

    # relative translation: express (xyz_cur - xyz_ref) in reference yaw frame
    xyz_rel = r_ref_yaw.inv().apply(xyz_cur - xyz_ref)

    return np.concatenate([xyz_rel, quat_rel], axis=-1)


def compute_imu_relative(reference_pose, current_pose):
    """
    Compute current_pose relative to reference_pose.

    Args:
        reference_pose: (N, 4), each row = [qw, qx, qy, qz] in world frame
        current_pose:   (N, 4), same format

    Returns:
        (N, 4): relative pose where
            - quaternion is ref_yaw^{-1} * current (only yaw of reference is removed)
    """
    if reference_pose.shape != current_pose.shape:
        raise ValueError(
            f"Cannot compute relative pose: "
            f"Pose dimensions don't match ({reference_pose.shape} vs {current_pose.shape})"
        )
    
    # scipy's Rotation expects quaternions in xyzw order
    def wxyz_to_xyzw(q):
        return np.roll(q, -1, axis=-1)

    def xyzw_to_wxyz(q):
        return np.roll(q, 1, axis=-1)

    quat_ref = normalize_quaternion(reference_pose)
    quat_cur = normalize_quaternion(current_pose)

    r_ref = Rotation.from_quat(wxyz_to_xyzw(quat_ref))
    r_cur = Rotation.from_quat(wxyz_to_xyzw(quat_cur))

    # Extract only the yaw component of reference rotation (rotation around Z axis)
    # Use arctan2 on the rotation matrix to avoid gimbal lock
    R_ref = r_ref.as_matrix()  # (N, 3, 3)
    yaw_ref = np.arctan2(R_ref[:, 1, 0], R_ref[:, 0, 0])  # (N,)
    r_ref_yaw = Rotation.from_euler("z", yaw_ref[:, None])  # yaw-only rotation

    # relative rotation: ref_yaw^{-1} * current
    r_relative = r_ref_yaw.inv() * r_cur
    quat_rel = xyzw_to_wxyz(r_relative.as_quat())

    return quat_rel


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Args:
        quat: shape (..., 4); the last dim is [w, x, y, z]

    Returns:
        Rotation matrix with shape (..., 3, 3)
    """
    orig_shape = quat.shape
    flat = quat.reshape(-1, 4)
    quat_xyzw = flat[:, [1, 2, 3, 0]]
    # scipy expects quaternions in [x, y, z, w] order.
    rotmat = Rotation.from_quat(quat_xyzw).as_matrix()
    return rotmat.reshape(*orig_shape[:-1], 3, 3)


def _rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
    """
    Args:
        rotmat: shape (..., 3, 3)

    Returns:
        Quaternion with shape (..., 4) in [w, x, y, z] order
    """
    orig_shape = rotmat.shape
    flat = rotmat.reshape(-1, 3, 3)
    quat_xyzw = Rotation.from_matrix(flat).as_quat()
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    return quat_wxyz.reshape(*orig_shape[:-2], 4)


def _sixd_to_rotmat(rot_3x2: np.ndarray) -> np.ndarray:
    """
    Args:
        rot_3x2: shape (..., 3, 2), first two columns of a 6D rotation representation

    Returns:
        Rotation matrix with shape (..., 3, 3)
    """
    orig_shape = rot_3x2.shape
    flat = rot_3x2.reshape(-1, 3, 2)
    r1 = flat[..., :, 0]
    r2 = flat[..., :, 1]

    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = r2 - np.sum(r2 * r1, axis=-1, keepdims=True) * r1
    r2 = r2 / np.linalg.norm(r2, axis=-1, keepdims=True)
    r3 = np.cross(r1, r2, axis=-1)

    rotmat = np.stack([r1, r2, r3], axis=-1)
    return rotmat.reshape(*orig_shape[:-2], 3, 3)


def quaternion_to_rotation_6d(data: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to 6D rotation representations.

    Args:
        data: (N, 15, 7) / (N, 4), 7 is xyz + quaternion [w, x, y, z], 4 is quaternion only

    Returns:
        (N, 15, 9) / (N, 6), 9 is xyz + 6D rotation, 6 is 6D rotation only
    """
    if data.ndim == 3:
        xyz = data[..., :3]                                  # numpy (N, 15, 3)
        rot_3x3 = _quat_to_rotmat(data[..., 3:])             # numpy (N, 15, 3, 3)
        rot_3x2 = rot_3x3[..., :2].transpose(0, 1, 3, 2).reshape(-1, 15, 6)  # numpy (N, 15, 6), column-major
        new_data = np.concatenate([xyz, rot_3x2], axis=-1)   # numpy (N, 15, 9)
    elif data.ndim == 2:
        rot_3x3 = _quat_to_rotmat(data)                      # numpy (N, 3, 3)
        rot_3x2 = rot_3x3[..., :2].transpose(0, 2, 1).reshape(-1, 6)  # numpy (N, 6), column-major
        new_data = rot_3x2
    else:
        raise ValueError(f"Expected data dimension 2 or 3, got {data.ndim}")
    return new_data


def rotation_6d_to_quaternion(data: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to quaternions.

    Args:
        data: (N, 15, 9) / (N, 6), 9 is xyz + 6D rotation, 6 is 6D rotation only

    Returns:
        (N, 15, 7) / (N, 4), 7 is xyz + quaternion [w, x, y, z], 4 is quaternion only
    """
    if data.ndim == 3:
        num_joints = data.shape[1]
        xyz = data[..., :3]                                              # (N, 15, 3)
        rot_3x2 = data[..., 3:].reshape(-1, num_joints, 2, 3).transpose(0, 1, 3, 2)  # (N, 15, 3, 2), column-major
        rot_3x3 = _sixd_to_rotmat(rot_3x2)                               # (N, 15, 3, 3)
        quat = _rotmat_to_quat(rot_3x3)                                  # (N, 15, 4)
        new_data = np.concatenate([xyz, quat], axis=-1)                  # (N, 15, 7)
    elif data.ndim == 2:
        rot_3x2 = data.reshape(-1, 2, 3).transpose(0, 2, 1)              # (N, 3, 2), column-major
        rot_3x3 = _sixd_to_rotmat(rot_3x2)                               # (N, 3, 3)
        new_data = _rotmat_to_quat(rot_3x3)                              # (N, 4)
    else:
        raise ValueError(f"Expected data dimension 2 or 3, got {data.ndim}")

    return new_data


def mocap_to_root_relative(data: np.ndarray) -> np.ndarray:
    """
    Convert (T, 15, 9) mocap to [root_delta_xyz, root-relative mocap].

    root_delta[t] = root_xyz[t + 1] - root_xyz[t], with the last delta set to 0.
    Mocap xyz values are root-relative; 6D rotations are unchanged.

    Args:
        data: (T, 15, 9), each point is [xyz(3), rot_6d(6)].

    Returns:
        (T, 138): [root_delta_xyz(3), flattened 15x9 mocap(135)].
    """
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[1:] != (15, 9):
        raise ValueError(f"Expected data shape (T, 15, 9), got {data.shape}")

    mocap_data = data.copy()
    num_frames, num_joints, pose_dim = mocap_data.shape

    xyz = mocap_data[:, :, :3]
    root_delta = np.zeros((num_frames, 3), dtype=mocap_data.dtype)
    if num_frames > 1:
        root_delta[:-1] = xyz[1:, 0, :] - xyz[:-1, 0, :]

    mocap_data[:, :, :3] = xyz - xyz[:, :1, :]
    mocap_data_flat = mocap_data.reshape(num_frames, num_joints * pose_dim)

    return np.concatenate([root_delta, mocap_data_flat], axis=1)


def mocap_to_root_relative_delta(data: np.ndarray) -> np.ndarray:
    """Convert mocap xyz to root-relative forward deltas.

    The output layout stays identical to :func:`mocap_to_root_relative`, but
    every joint xyz at index ``t`` is
    ``root_relative_xyz[t + 1] - root_relative_xyz[t]``. The final xyz delta is
    zero, matching the existing root-delta boundary convention. Rotations are
    kept unchanged at their original frame.
    """
    converted = mocap_to_root_relative(data)
    mocap_data = converted[:, 3:].reshape(-1, 15, 9).copy()
    root_relative_xyz = mocap_data[:, :, :3].copy()

    mocap_data[:, :, :3] = 0.0
    if len(mocap_data) > 1:
        mocap_data[:-1, :, :3] = root_relative_xyz[1:] - root_relative_xyz[:-1]

    converted[:, 3:] = mocap_data.reshape(len(mocap_data), -1)
    return converted


def restore_mocap_from_root_relative(
    mocap_root_rel: np.ndarray,
    init_root_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """
    Restore [root_delta_xyz, root-relative 11x9 mocap] to absolute 11x9 mocap.

    Uses the same delta convention as mocap_to_root_relative: delta[t] points from
    frame t to frame t + 1, and the last delta is ignored during restoration.

    Args:
        mocap_root_rel: (T, 102), [root_delta_xyz(3), flattened 11x9 mocap(99)].
        init_root_xyz: Optional (3,) root xyz for the first frame. Defaults to zeros.

    Returns:
        (T, 11, 9) mocap with absolute xyz positions.
    """
    mocap_root_rel = np.asarray(mocap_root_rel)
    if mocap_root_rel.ndim != 2:
        raise ValueError(f"Expected mocap_root_rel shape (T, 102), got {mocap_root_rel.shape}")

    if mocap_root_rel.shape[1] != 102:
        raise ValueError(f"Expected mocap_root_rel dim 102, got {mocap_root_rel.shape[1]}")

    if init_root_xyz is None:
        init_root_xyz = np.zeros(3, dtype=mocap_root_rel.dtype)
    else:
        init_root_xyz = np.asarray(init_root_xyz, dtype=mocap_root_rel.dtype)
        if init_root_xyz.shape != (3,):
            raise ValueError(f"Expected init_root_xyz shape (3,), got {init_root_xyz.shape}")

    root_delta = mocap_root_rel[:, :3]
    mocap_data = mocap_root_rel[:, 3:].reshape(-1, 11, 9).copy()
    mocap_data[:, 0, :3] = 0.0

    num_frames = mocap_root_rel.shape[0]
    root_xyz = np.zeros((num_frames, 3), dtype=mocap_root_rel.dtype)

    if num_frames > 0:
        root_xyz[0] = init_root_xyz
    if num_frames > 1:
        root_xyz[1:] = init_root_xyz[None, :] + np.cumsum(root_delta[:-1], axis=0)
    next_init_root_xyz = root_xyz[-1] + root_delta[-1]

    mocap_data[:, :, :3] = mocap_data[:, :, :3] + root_xyz[:, None, :]
    return mocap_data, next_init_root_xyz


def restore_mocap_from_root_relative_delta(
    mocap_root_rel_delta: np.ndarray,
    init_root_xyz: np.ndarray | None = None,
    init_joint_relative_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore root and joint-relative XYZ *deltas* to absolute mocap poses.

    The input layout is ``[root_delta_xyz(3), flattened 11 x 9 mocap]``.  The
    XYZ of each mocap point is a frame-to-frame delta in the root frame; its
    6D rotation remains an absolute rotation representation.  Consequently,
    this is the inverse of :func:`mocap_to_root_relative_delta` after the
    dataset has selected its 11 retained points.

    The delta at step ``t`` moves the pose from step ``t`` to ``t + 1``.  This
    follows the existing root-delta convention: the first restored frame uses
    the supplied initial pose and the final delta only determines the next
    chunk's initial state.
    """
    mocap_root_rel_delta = np.asarray(mocap_root_rel_delta)
    if mocap_root_rel_delta.ndim != 2 or mocap_root_rel_delta.shape[1] != 102:
        raise ValueError(
            "Expected mocap_root_rel_delta shape (T, 102), got "
            f"{mocap_root_rel_delta.shape}"
        )

    dtype = mocap_root_rel_delta.dtype
    if init_root_xyz is None:
        init_root_xyz = np.zeros(3, dtype=dtype)
    else:
        init_root_xyz = np.asarray(init_root_xyz, dtype=dtype)
        if init_root_xyz.shape != (3,):
            raise ValueError(f"Expected init_root_xyz shape (3,), got {init_root_xyz.shape}")

    if init_joint_relative_xyz is None:
        init_joint_relative_xyz = np.zeros((11, 3), dtype=dtype)
    else:
        init_joint_relative_xyz = np.asarray(init_joint_relative_xyz, dtype=dtype)
        if init_joint_relative_xyz.shape != (11, 3):
            raise ValueError(
                "Expected init_joint_relative_xyz shape (11, 3), got "
                f"{init_joint_relative_xyz.shape}"
            )

    root_delta = mocap_root_rel_delta[:, :3]
    mocap_data = mocap_root_rel_delta[:, 3:].reshape(-1, 11, 9).copy()
    # Keep the original deltas for the continuation state below.  ``mocap_data``
    # is overwritten with absolute XYZ before returning, so a view here would
    # otherwise be silently changed to absolute positions.
    joint_relative_delta = mocap_data[:, :, :3].copy()
    num_frames = mocap_root_rel_delta.shape[0]

    root_xyz = np.zeros((num_frames, 3), dtype=dtype)
    joint_relative_xyz = np.zeros((num_frames, 11, 3), dtype=dtype)
    if num_frames > 0:
        root_xyz[0] = init_root_xyz
        joint_relative_xyz[0] = init_joint_relative_xyz
    if num_frames > 1:
        root_xyz[1:] = init_root_xyz[None, :] + np.cumsum(root_delta[:-1], axis=0)
        joint_relative_xyz[1:] = (
            init_joint_relative_xyz[None, :, :]
            + np.cumsum(joint_relative_delta[:-1], axis=0)
        )

    mocap_data[:, :, :3] = joint_relative_xyz + root_xyz[:, None, :]
    next_root_xyz = root_xyz[-1] + root_delta[-1] if num_frames else init_root_xyz.copy()
    next_joint_relative_xyz = (
        joint_relative_xyz[-1] + joint_relative_delta[-1]
        if num_frames
        else init_joint_relative_xyz.copy()
    )
    return mocap_data, next_root_xyz, next_joint_relative_xyz
