# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility methods for the Ant environments.

Adapted from rllab maze_env_utils.py.
"""
import math


class Move(object):
    """Movable attributes."""

    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17
    SpinXY = 18


def can_move_x(movable):
    """TODO.

    Parameters
    ----------
    movable : int
        TODO

    Returns
    -------
    bool
        TODO
    """
    return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ, Move.SpinXY]


def can_move_y(movable):
    """TODO.

    Parameters
    ----------
    movable : int
        TODO

    Returns
    -------
    bool
        TODO
    """
    return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ, Move.SpinXY]


def can_move_z(movable):
    """TODO.

    Parameters
    ----------
    movable : int
        TODO

    Returns
    -------
    bool
        TODO
    """
    return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_spin(movable):
    """TODO.

    Parameters
    ----------
    movable : int
        TODO

    Returns
    -------
    bool
        TODO
    """
    return movable in [Move.SpinXY]


def can_move(movable):
    """TODO.

    Parameters
    ----------
    movable : int
        TODO

    Returns
    -------
    bool
        TODO
    """
    return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def construct_maze(maze_id='Maze'):
    """Construct the structure of the maze.

    Parameters
    ----------
    maze_id : str
        type of environment. This dictates the structure the maze will adopt.

    Returns
    -------
    list of Any
        TODO
    """
    if maze_id == 'Maze':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'Push':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 'r', 1, 1],
            [1, 0, Move.XY, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'Fall':
        structure = [
            [1, 1, 1, 1],
            [1, 'r', 0, 1],
            [1, 0, Move.YZ, 1],
            [1, -1, -1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    elif maze_id == 'Block':
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'r', 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == 'BlockMaze':
        structure = [
            [1, 1, 1, 1],
            [1, 'r', 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    else:
        raise NotImplementedError(
            'The provided MazeId %s is not recognized' % maze_id)

    return structure


def line_intersect(pt1, pt2, ptA, ptB):
    """Return the intersection of Line(pt1,pt2) and Line(ptA,ptB).

    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html

    Parameters
    ----------
    pt1 : TODO
        TODO
    pt2 : TODO
        TODO
    ptA : TODO
        TODO
    ptB : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0, 0, 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return xi, yi, 1, r, s


def ray_segment_intersect(ray, segment):
    """Check ray intersection with segment.

    Check if the ray originated from (x, y) with direction theta intersects the
    line segment (x1, y1) -- (x2, y2), and return the intersection point if
    there is one

    Parameters
    ----------
    ray : TODO
        TODO
    segment : TODO
        TODO

    Returns
    -------
    TODO
        TODO
    """
    (x, y), theta = ray
    # (x1, y1), (x2, y2) = segment
    pt1 = (x, y)
    len = 1
    pt2 = (x + len * math.cos(theta), y + len * math.sin(theta))
    xo, yo, valid, r, s = line_intersect(pt1, pt2, *segment)
    if valid and r >= 0 and 0 <= s <= 1:
        return xo, yo
    return None


def point_distance(p1, p2):
    """Return the distance between two points.

    Parameters
    ----------
    p1 : (float, float)
        (x,y) values of point 1
    p2 : (float, float)
        (x,y) values of point 2

    Returns
    -------
    float
        the distance between the two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
