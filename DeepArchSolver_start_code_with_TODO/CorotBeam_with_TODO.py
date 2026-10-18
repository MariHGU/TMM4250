# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:43:51 2018

@author: bjohau
"""

import numpy as np
import math

def rot_matrix(theta):
    """
    Return the 2x2 rotation matrix representing a rotation theta
    :param theta:  rotation angle in radians
    :return: Rotation matrix (or tensor)
    """
    s = math.sin(theta)
    c = math.cos(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return R

def beam2local_def_disp(ex,ey, disp_global):
    """

    :param ex: element x coordinate [x1, x2] in undeformed position
    :param ey: element y coordinate [y1, y2] in undeformed position
    :param disp_global:  displacement vector [u1, v1, r1, u2, v2, r2] in global directions
    :return: disp_local_def: displacement vector [u1, v1, r1, u2, v2, r2] in local directions
    """
    eVec12 = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    L0 = math.sqrt(eVec12 @ eVec12)

    #Added: Position of end nodes
    x1_now = ex[0] + disp_global[0]
    y1_now = ey[0] + disp_global[1]
    x2_now = ex[1] + disp_global[3]
    y2_now = ey[1] + disp_global[4]

    #Added: Length between nodes
    dx = x2_now - x1_now
    dy = y2_now - y1_now
    Ld = math.sqrt(dx**2 + dy**2)

    #Added: Chord rotation
    angle_now = math.atan2(dy, dx)
    angle_start = math.atan2(eVec12[1], eVec12[0])
    chord_rotation = angle_now - angle_start

    #Added: Node rotations
    theta1_def = disp_global[2] - chord_rotation
    theta2_def = disp_global[5] - chord_rotation

    #Added: To keep angles between -pi and +pi
    theta1_def = math.atan2(math.sin(theta1_def), math.cos(theta1_def))
    theta2_def = math.atan2(math.sin(theta2_def), math.cos(theta2_def))

    def_disp_local = np.array([ -0.5*(Ld - L0),
                                0.0,
                                theta1_def,
                                0.5 * (Ld - L0),
                                0.0,
                                theta2_def])
    return def_disp_local



def beam2corot_Ke_and_Fe(ex,ey,ep, disp_global):
    """
    Compute the stiffness matrix and internal forces for a two dimensional beam element
    relative to deformed configuration.

    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
    :param list disp_global displacement vector for the element [tx1,ty1,rz1,tx2,ty2,rz2]

    :return mat Ke_global: element stiffness matrix [6 x 6]
    :return mat fe_int_global: element internal forces [6 x 1]
    """
    # Undeformed length and unit vector along element
    eVec12 = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    L0 = math.sqrt(eVec12 @ eVec12)
    eVec12 /= L0

    # Deformed position and unit vector along element
    ex_def = ex + [disp_global[0], disp_global[3]]
    ey_def = ey + [disp_global[1], disp_global[4]]

    #Added: Local deformation
    def_disp_local = beam2local_def_disp(ex, ey, disp_global)

    #Added: Local forces
    Kle = beam2local_stiff(L0, ep)
    fe_int_local = Kle @ def_disp_local

    #Added: Transformation from local to global
    Te = beam2corot_Te(ex_def, ey_def)

    # Problem 2: Geometric stiffness
    # Current element length and orientation
    d = np.array([ex_def[1] - ex_def[0],
                  ey_def[1] - ey_def[0]])
    L = np.linalg.norm(d)

    n = d / L
    t = np.array([-n[1], n[0]])

    # Generalized deformations: axial extension and end rotations
    q = np.array([
        2.0 * def_disp_local[3],
        def_disp_local[2],
        def_disp_local[5]
    ])

    E, A, I = ep

    # Material stiffness in generalized coordinates
    D = np.array([
        [E*A/L0,       0,          0],
        [0,       4*E*I/L0,   2*E*I/L0],
        [0,       2*E*I/L0,   4*E*I/L0]
    ])

    # Internal axial force and end moments
    p = D @ q
    N, M1, M2 = p

    # Transformation from global DOFs to generalized deformations
    B = np.array([
        [-n[0], -n[1], 0, n[0], n[1], 0],
        [t[0]/L, t[1]/L, 1, -t[0]/L, -t[1]/L, 0],
        [t[0]/L, t[1]/L, 0, -t[0]/L, -t[1]/L, 1]
    ])

    # Internal forces in global coordinates
    fe_int_global = B.T @ p

    # Material contribution to tangent stiffness
    K_material = B.T @ D @ B

    # Geometric contribution to tangent stiffness
    J = np.array([
        [-1, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 1, 0]
    ])

    K_geo_2d = (
        (N / L) * np.outer(t, t)
        + ((M1 + M2) / L**2)
        * (np.outer(t, n) + np.outer(n, t))
    )

    K_geometric = J.T @ K_geo_2d @ J

    # Total tangent stiffness
    Ke_global = K_material + K_geometric

    return Ke_global, fe_int_global


    
def beam2corot_Te(ex,ey):
    """
    Compute the transformation matrix for an element
    
    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia   
    :param list eq: distributed loads, local directions [qx, qy]
    :return mat Te: element transformation from global to local
    """

    n = np.array([ex[1]-ex[0],ey[1]-ey[0]])
    L = np.linalg.norm(n)
    n = n / L  
    
    Te=np.array([
        [ n[0], n[1],  0.,    0.,    0.,   0.],
        [-n[1], n[0],  0.,    0.,    0.,   0.],
        [0.,    0.,    1.,    0.,    0.,   0.],
        [0.,    0.,    0.,   n[0],  n[1],  0.],
        [0.,    0.,    0.,  -n[1],  n[0],  0.],
        [0.,    0.,    0.,    0.,    0.,   1.]
    ])
    

    return Te
    
    
def beam2local_stiff(L,ep):
    """
    Compute the stiffness matrix for a two dimensional beam element.
    
    :param list L : element length
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia   
    :return mat Kle: element stiffness matrix [6 x 6]
    """
    
    E=ep[0]
    A=ep[1]
    I=ep[2]
        
    Kle = np.array([
        [E*A/L,              0.,           0.,    -E*A/L,            0.,           0. ],
        [   0.,    12*E*I/L**3.,  6*E*I/L**2.,        0., -12*E*I/L**3.,  6*E*I/L**2. ],
        [   0.,     6*E*I/L**2.,      4*E*I/L,        0.,  -6*E*I/L**2.,     2*E*I/L  ],
        [-E*A/L,             0.,           0.,     E*A/L,            0.,           0. ],
        [   0.,   -12*E*I/L**3., -6*E*I/L**2.,        0.,  12*E*I/L**3., -6*E*I/L**2. ],
        [   0.,     6*E*I/L**2.,      2*E*I/L,        0.,  -6*E*I/L**2.,      4*E*I/L ]
    ])
     
    return Kle


def beam2e(ex, ey, ep, eq=None):
    """
    Compute the linear stiffness matrix for a two dimensional beam element.
    Largely from CALFEM core module

    :param list ex: element x coordinates [x1, x2]
    :param list ey: element y coordinates [y1, y2]
    :param list ep: element properties [E, A, I], E - Young's modulus, A - Cross section area, I - Moment of inertia
    :param list eq: distributed loads, local directions [qx, qy]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: element consistent force for distributed load [6 x 1] (if eq!=None)
    """

    n = np.array([ex[1] - ex[0], ey[1] - ey[0]])
    L = np.linalg.norm(n)
    n = n / L

    qx = 0.
    qy = 0.
    if not eq is None:
        qx = eq[0]
        qy = eq[1]

    Kle = beam2local_stiff(L,ep)

    fle = L * np.array([[qx / 2, qy / 2, qy * L / 12, qx / 2, qy / 2, -qy * L / 12]]).T

    Te = beam2corot_Te(ex,ey)

    Ke = Te.T @ Kle @ Te
    fe = Te.T @ fle

    if eq is None:
        return Ke
    else:
        return Ke, fe