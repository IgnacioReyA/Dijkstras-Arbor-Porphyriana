import numpy as np
import math

def barycentric_coords(point, triangle_vertices):
    """
    Calculate barycentric coordinates for a point within a triangle (using XY only).
    
    Args:
        point: (x, y) coordinates of the point
        triangle_vertices: list of 3 vertices, each with (x, y, z) coordinates
    
    Returns:
        tuple: (u, v, w) barycentric coordinates
    """
    # Extract XY coordinates only
    p = np.array([point[0], point[1]])
    v0 = np.array([triangle_vertices[0][0], triangle_vertices[0][1]])
    v1 = np.array([triangle_vertices[1][0], triangle_vertices[1][1]])
    v2 = np.array([triangle_vertices[2][0], triangle_vertices[2][1]])
    
    # Calculate vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    
    # Calculate dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)
    
    # Calculate barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1 - u - v
    
    return u, v, w


def interpolate_height(point, triangle_vertices):
    """
    Interpolate the height (z) of a point inside a triangle using barycentric coordinates.
    """
    u, v, w = barycentric_coords(point, triangle_vertices)
    z = u * triangle_vertices[0][2] + v * triangle_vertices[1][2] + w * triangle_vertices[2][2]
    return z


def face_slope(triangle_normal):
    """
    Calculate face slope from triangle normal.
    Formula: slope = arctan(√(n_x^2 + n_y^2) / |n_z|) in degrees.
    """
    n_x, n_y, n_z = triangle_normal
    numerator = math.sqrt(n_x**2 + n_y**2)
    denominator = abs(n_z)
    
    if denominator == 0:
        return 90.0  # Vertical face
    
    slope_radians = math.atan(numerator / denominator)
    return math.degrees(slope_radians)


def aspect(triangle_normal):
    """
    Calculate aspect from triangle normal.
    Formula: atan2(n_y, n_x) in degrees.
    """
    n_x, n_y, n_z = triangle_normal
    aspect_radians = math.atan2(n_y, n_x)
    return math.degrees(aspect_radians)


def point_in_triangle(point, triangle_vertices):
    """
    Test if a point is inside a triangle using barycentric coordinates (XY only).
    """
    u, v, w = barycentric_coords(point, triangle_vertices)
    return u >= 0 and v >= 0 and w >= 0
