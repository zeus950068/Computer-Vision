###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # 步驟1: 奇異值分解（SVD）
    U, D, Vt = np.linalg.svd(E)

    # 定義 W 和 Z 矩陣
    W = np.array([[0, -1, 0], 
                  [1, 0, 0],
                  [0, 0, 1]])
    
    Z = np.array([[0, 1, 0], 
                  [-1, 0, 0], 
                  [0, 0, 0]])
    
    # 計算旋轉矩陣 R
    Q1 = U @ W @ Vt
    Q2 = U @ W.T @ Vt
    # print("Matrix Q1:\n", Q1)
    # print("Matrix Q2:\n", Q2)
    det_Q1 = np.linalg.det(Q1)
    det_Q2 = np.linalg.det(Q2)
    R1 = det_Q1 * Q1
    R2 = det_Q2 * Q2
    # print("R1:\n", R1)            # shape:(3, 3)
    # print("R2:\n", R2)            # shape:(3, 3)

    # 步驟2: 估算平移向量 T
    u3 = U[:, 2]
    T1 = u3                         # shape:(3,)
    T2 = -u3                        # shape:(3,)
    T1_add = T1[:, np.newaxis]      # shape:(3,1)
    T2_add = T2[:, np.newaxis]      # shape:(3,1)

    # 步驟3: 返回四種可能的組合
    RT1 = np.hstack((R1, T1_add))
    RT2 = np.hstack((R1, T2_add))
    RT3 = np.hstack((R2, T1_add))
    RT4 = np.hstack((R2, T2_add))
    
    return np.array([RT1, RT2, RT3, RT4])

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    M = len(image_points)
    A = np.zeros((2 * M, 4))
    # print(image_points.shape)
    # print(image_points)
    # print(camera_matrices.shape)
    # print(camera_matrices)
    # print(camera_matrices[0][2, 0])
    # print("")
    for i in range(M):
        A[2 * i, 0] = image_points[i][1] * camera_matrices[i][2, 0] - camera_matrices[i][1, 0]
        A[2 * i, 1] = image_points[i][1] * camera_matrices[i][2, 1] - camera_matrices[i][1, 1]
        A[2 * i, 2] = image_points[i][1] * camera_matrices[i][2, 2] - camera_matrices[i][1, 2]
        A[2 * i, 3] = image_points[i][1] * camera_matrices[i][2, 3] - camera_matrices[i][1, 3]
        # print('A', A)
        A[2 * i + 1, 0] = camera_matrices[i][0, 0] - image_points[i][0] * camera_matrices[i][2, 0]
        A[2 * i + 1, 1] = camera_matrices[i][0, 1] - image_points[i][0] * camera_matrices[i][2, 1]
        A[2 * i + 1, 2] = camera_matrices[i][0, 2] - image_points[i][0] * camera_matrices[i][2, 2]
        A[2 * i + 1, 3] = camera_matrices[i][0, 3] - image_points[i][0] * camera_matrices[i][2, 3]
        # print('A', A)
        

    # Perform SVD
    _, _, Vt = np.linalg.svd(A)
    # print(Vt)
    # print(Vt[-1, :-1])
    # print(Vt[-1, -1])
    # print("")

    # The 3D point is the last column of Vt
    point_3d = Vt[-1, :-1] / Vt[-1, -1]
    # print(point_3d)
    # print("")

    return point_3d

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    num_images = len(camera_matrices)   # len(camera_matrices) = 2
    error = np.array([])

    for i in range(num_images):
        Mi = camera_matrices[i]
        u, v = image_points[i]

        # 3D location of a point 3D點的坐標
        P = np.hstack((point_3d, 1))
        # print(matrix)
        # print("")

        # 將3D點P投影到圖像平面，計算重投影點
        projected_point = Mi @ P         # (3, 4)*(4, 1) = (3, 1)
        projected_u = projected_point[0] / projected_point[2]
        projected_v = projected_point[1] / projected_point[2]

        # 計算重投影誤差
        error = np.append(error, projected_u - u)
        error = np.append(error, projected_v - v)

    return error

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    num_views = len(camera_matrices)            # M = num_views = 2
    num_projections = 2 * num_views             # num_projections = 4
    jacobian = np.zeros((num_projections, 3))   # (4, 3)
    
    # Iterate through each view
    for i in range(num_views):
        Mi = camera_matrices[i]                 # (3, 4)
        # print(Mi)
        P = np.append(point_3d, 1)              # P = [X Y Z 1] (4, 1)
        y = Mi @ P                              # y = MiP
        # print("MiP:\n", y)
            
        jacobian[2 * i, :] = Mi[0, 0:3] / y[2] - Mi[2, 0:3] * y[0] / (y[2] ** 2)
        jacobian[2 * i + 1, :] = Mi[1, 0:3] / y[2] - Mi[2, 0:3] * y[1] / (y[2] ** 2)
    # print("(jacobian)jacobian result:\n", jacobian)
    return jacobian

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    point_3d = linear_estimate_3d_point(image_points, camera_matrices)
    for i in range(10):
        error = reprojection_error(point_3d, image_points, camera_matrices)
        # print("(nonlinear_estimate_3d_point)error:\n", error)
        
        jacobian_result = jacobian(point_3d, camera_matrices)
        # print("(nonlinear_estimate_3d_point)jacobian result:\n", jacobian_result)

        J_T = jacobian_result.T
        JTJ = np.dot(J_T, jacobian_result)
 
        J_inv = np.linalg.inv(JTJ)
        # print(point_3d)
        point_3d = point_3d - np.dot(np.dot(J_inv , J_T), error)
        # print("(iteraion ", i, ")", "point_3d: ", point_3d)
    return point_3d

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # Singular Value Decomposition of the Essential Matrix
    U, S, Vt = np.linalg.svd(E)
    
    # Calculate the rotation matrices R1 and R2
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    Z = np.array([[0, 1, 0], 
                [-1, 0, 0], 
                [0, 0, 0]])

    Q1 = U @ W @ Vt
    Q2 = U @ W.T @ Vt
    det_Q1 = np.linalg.det(Q1)
    det_Q2 = np.linalg.det(Q2)
    R1 = det_Q1 * Q1
    R2 = det_Q2 * Q2

    # # Calculate the translation vector
    # Tx = U @ Z @ U.T

    # # Calculate the Fundamention Matrix
    # K_inv = np.linalg.inv(K)
    # A = np.array([[1, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 0]])
    # F = K_inv.T @ Tx @ R2 @ K_inv
    # E1 = K.T @ F @ K
    # E = U @ A @ W @ U.T @ R1 
    
    u3 = U[:, 2]
    T1 = u3                         # shape:(3,)
    T2 = -u3                        # shape:(3,)
    T1_add = T1[:, np.newaxis]      # shape:(3,1)
    T2_add = T2[:, np.newaxis]      # shape:(3,1)
    
    RT = np.hstack((R2, T1_add))
    # projected_points = np.dot(K, RT @ image_points.reshape(4, 1))
    # z_coordinates = projected_points[2, :]
    # print(z_coordinates)
    
    # print(RT)
    return RT


if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E, np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)
    print("")

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length, fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')