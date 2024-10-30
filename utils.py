import torch
import numpy

def uniform(min_val, max_val, size):
    a = torch.rand(size, device=torch.device('cuda'))
    a = a * (max_val - min_val) + min_val
    return a

def np_uniform(min_val, max_val):
    a = numpy.random.rand()
    a = a * (max_val - min_val) + min_val
    return a

def scale(x, x_min, x_max, scale_min, scale_max):
    """
    Scales x in [x_min, x_max] to the range of [scale_min, scale_max].
    """
    return (x - x_min) / (x_max - x_min) * (scale_max - scale_min) + scale_min

def out_of_bounds(x, x_min, x_max):
    """
    Returns True if x is out of bounds.
    """
    return torch.logical_or((x < x_min).any(dim=1), (x > x_max).any(dim=1))

def normalize_angle(x):
    """Normalizes the angle to [0, 2pi].
    """
    angle = torch.fmod(x, 2 * torch.pi)
    negative_angle_indices = torch.where(angle < 0)[0]
    angle[negative_angle_indices] += 2 * torch.pi
    return angle

def angle_in_between(x, a, b):
    """Check whether angle x is in between angle a and b. Note that a is always smaller than b.
    """
    x = normalize_angle(x)
    a = normalize_angle(a)
    b = normalize_angle(b)
    return torch.logical_or(a >= x, x >= b)

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw angles to quaternion.
    """
    # Compute half angles
    cy = numpy.cos(yaw * 0.5)
    sy = numpy.sin(yaw * 0.5)
    cp = numpy.cos(pitch * 0.5)
    sp = numpy.sin(pitch * 0.5)
    cr = numpy.cos(roll * 0.5)
    sr = numpy.sin(roll * 0.5)

    # Compute quaternion components
    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return numpy.array([q_x, q_y, q_z, q_w])