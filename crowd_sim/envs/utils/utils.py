import numpy as np


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def counterclockwise(x1, y1, x2, y2, x3, y3):
    """
    Calculate if  point (x3, y3) lies in the left side of directed line segment from (x1, y1) to (x2, y2)

    """
    vec1 = np.array([x2 - x1, y2 - y1])
    vec2 = np.array([x3 - x1, y3 - y1])
    if np.cross(vec1, vec2) > 0:
        return True
    else:
        return False

def point_in_poly(px, py, vertex):
    """
    Calculate if  point (px, py) lies in the polygons represented by vertex (counterclockwise)
    """
    for i in range(len(vertex) - 1):
        p1_x = vertex[i][0]
        p1_y = vertex[i][1]
        p2_x = vertex[i+1][0]
        p2_y = vertex[i+1][1]
        if not counterclockwise(p1_x, p1_y, p2_x, p2_y, px, py):
            return False
    return True
